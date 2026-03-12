/*
 * LocalLLM.mm — React Native Turbo Module bridge for local-llm-rn (iOS).
 *
 * This is a thin Obj-C wrapper that converts between RN JS values and
 * the platform-neutral libhilum C API. All inference logic lives in
 * hilum_llm.cpp inside the engine.
 *
 * Platform-specific features (downloads, device capabilities, stream
 * cancellation) remain as RN-specific code.
 */

#import "LocalLLM.h"

#import <React/RCTBridge.h>
#import <React/RCTLog.h>

#include <string>
#include <unordered_map>
#include <mutex>
#include <atomic>

#include "../cpp/hilum/hilum_llm.h"

#import <Metal/Metal.h>
#import <os/proc.h>
#import <CommonCrypto/CommonDigest.h>

#ifdef RCT_NEW_ARCH_ENABLED
#import <memory>
#endif

// ── UUID generation ──────────────────────────────────────────────────────────

static NSString *generateUUID() {
  return [[NSUUID UUID] UUIDString];
}

// ── File path scoping ────────────────────────────────────────────────────────

static NSString *allowedStorageRoot() {
  static NSString *root = nil;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSApplicationSupportDirectory, NSUserDomainMask, YES);
    root = [paths.firstObject stringByAppendingPathComponent:@"local-llm"];
  });
  return root;
}

/// Returns YES if the path is under the allowed storage root (Application Support/local-llm/).
/// This prevents callers from reading/writing arbitrary filesystem locations.
static BOOL isPathAllowed(NSString *path) {
  NSString *resolved = [path stringByStandardizingPath];
  NSString *root = [allowedStorageRoot() stringByStandardizingPath];
  return [resolved hasPrefix:root];
}

// ── Handle maps ──────────────────────────────────────────────────────────────

static std::mutex g_mutex;

static std::unordered_map<std::string, hilum_model *> g_models;
static std::unordered_map<std::string, hilum_context *> g_contexts;
static std::unordered_map<std::string, hilum_mtmd *> g_mtmd_contexts;
static std::unordered_map<std::string, hilum_emb_ctx *> g_emb_contexts;

// ── Log state ────────────────────────────────────────────────────────────────

static std::atomic<bool> g_log_events_enabled{false};
static __weak LocalLLM *g_log_module = nil;

// ── Gen params helper ────────────────────────────────────────────────────────

static hilum_gen_params parse_gen_params(NSDictionary *options) {
  hilum_gen_params p = hilum_gen_default_params();
  if (options[@"max_tokens"])         p.max_tokens = [options[@"max_tokens"] intValue];
  if (options[@"temperature"])        p.temperature = [options[@"temperature"] floatValue];
  if (options[@"top_p"])              p.top_p = [options[@"top_p"] floatValue];
  if (options[@"top_k"])              p.top_k = [options[@"top_k"] intValue];
  if (options[@"repeat_penalty"])     p.repeat_penalty = [options[@"repeat_penalty"] floatValue];
  if (options[@"frequency_penalty"])  p.frequency_penalty = [options[@"frequency_penalty"] floatValue];
  if (options[@"presence_penalty"])   p.presence_penalty = [options[@"presence_penalty"] floatValue];
  if (options[@"seed"])               p.seed = [options[@"seed"] intValue];
  if (options[@"n_past"])             p.n_past = [options[@"n_past"] intValue];
  return p;
}

// Keep grammar strings alive alongside params
struct GenContext {
  hilum_gen_params params;
  std::string grammar;
  std::string grammar_root;

  void finalize() {
    params.grammar      = grammar.empty()      ? nullptr : grammar.c_str();
    params.grammar_root  = grammar_root.empty() ? nullptr : grammar_root.c_str();
  }
};

static GenContext parse_gen_context(NSDictionary *options) {
  GenContext gc;
  gc.params = parse_gen_params(options);
  if (options[@"grammar"])      gc.grammar = [options[@"grammar"] UTF8String];
  if (options[@"grammar_root"]) gc.grammar_root = [options[@"grammar_root"] UTF8String];
  gc.finalize();
  return gc;
}

// ── Base64 decoding ──────────────────────────────────────────────────────────

static std::vector<uint8_t> decode_base64(NSString *base64) {
  NSData *data = [[NSData alloc] initWithBase64EncodedString:base64 options:0];
  if (!data) return {};
  const uint8_t *bytes = (const uint8_t *)[data bytes];
  return std::vector<uint8_t>(bytes, bytes + [data length]);
}

// ── Inference dispatch queue ─────────────────────────────────────────────────

static dispatch_queue_t inference_queue() {
  static dispatch_queue_t q = dispatch_queue_create("com.hilum.llm.inference", DISPATCH_QUEUE_SERIAL);
  return q;
}

// ── Download session management ──────────────────────────────────────────────

@interface LLMDownloadDelegate : NSObject <NSURLSessionDownloadDelegate>
@property (nonatomic, weak) LocalLLM *module;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSString *> *destPaths;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSData *> *resumeData;
@end

@implementation LLMDownloadDelegate

- (instancetype)initWithModule:(LocalLLM *)module {
  self = [super init];
  if (self) {
    _module = module;
    _destPaths = [NSMutableDictionary new];
    _resumeData = [NSMutableDictionary new];
  }
  return self;
}

- (void)URLSession:(NSURLSession *)session
      downloadTask:(NSURLSessionDownloadTask *)downloadTask
      didWriteData:(int64_t)bytesWritten
 totalBytesWritten:(int64_t)totalBytesWritten
totalBytesExpectedToWrite:(int64_t)totalBytesExpectedToWrite {
  NSString *url = downloadTask.originalRequest.URL.absoluteString;
  double percent = totalBytesExpectedToWrite > 0
    ? (double)totalBytesWritten / (double)totalBytesExpectedToWrite * 100.0
    : 0.0;
  [_module sendEventWithName:@"onDownloadProgress" body:@{
    @"url": url ?: @"",
    @"downloaded": @(totalBytesWritten),
    @"total": @(totalBytesExpectedToWrite),
    @"percent": @(percent),
  }];
}

- (void)URLSession:(NSURLSession *)session
      downloadTask:(NSURLSessionDownloadTask *)downloadTask
didFinishDownloadingToURL:(NSURL *)location {
  NSString *url = downloadTask.originalRequest.URL.absoluteString;
  NSString *destPath = _destPaths[url];
  if (destPath) {
    NSError *error = nil;
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm removeItemAtPath:destPath error:nil];
    [fm createDirectoryAtPath:[destPath stringByDeletingLastPathComponent]
      withIntermediateDirectories:YES attributes:nil error:nil];
    [fm moveItemAtURL:location toURL:[NSURL fileURLWithPath:destPath] error:&error];
    if (error) {
      [_module sendEventWithName:@"onDownloadError" body:@{
        @"url": url ?: @"",
        @"error": error.localizedDescription ?: @"Move failed",
        @"resumable": @NO,
      }];
      return;
    }
  }
  [_module sendEventWithName:@"onDownloadComplete" body:@{
    @"url": url ?: @"",
  }];
}

- (void)URLSession:(NSURLSession *)session
              task:(NSURLSessionTask *)task
didCompleteWithError:(NSError *)error {
  if (!error) return;
  NSString *url = task.originalRequest.URL.absoluteString;
  NSData *data = error.userInfo[NSURLSessionDownloadTaskResumeData];
  BOOL resumable = data != nil;
  if (resumable && url) {
    _resumeData[url] = data;
  }
  [_module sendEventWithName:@"onDownloadError" body:@{
    @"url": url ?: @"",
    @"error": error.localizedDescription ?: @"Download failed",
    @"resumable": @(resumable),
  }];
}

@end

// ── Module implementation ────────────────────────────────────────────────────

@implementation LocalLLM {
  NSURLSession *_downloadSession;
  LLMDownloadDelegate *_downloadDelegate;
  bool _hasListeners;
}

#ifdef RCT_NEW_ARCH_ENABLED
- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params
{
  return std::make_shared<facebook::react::NativeLocalLLMSpecJSI>(params);
}
#endif

RCT_EXPORT_MODULE()

+ (BOOL)requiresMainQueueSetup {
  return NO;
}

- (instancetype)init {
  self = [super init];
  if (self) {
    _hasListeners = NO;
    _downloadDelegate = [[LLMDownloadDelegate alloc] initWithModule:self];
    NSURLSessionConfiguration *config =
      [NSURLSessionConfiguration backgroundSessionConfigurationWithIdentifier:@"com.hilum.llm.downloads"];
    config.sessionSendsLaunchEvents = YES;
    _downloadSession = [NSURLSession sessionWithConfiguration:config
                                                    delegate:_downloadDelegate
                                               delegateQueue:nil];
  }
  return self;
}

- (NSArray<NSString *> *)supportedEvents {
  return @[
    @"onToken",
    @"onBatchToken",
    @"onQuantizeComplete",
    @"onLog",
    @"onDownloadProgress",
    @"onDownloadComplete",
    @"onDownloadError",
  ];
}

- (void)startObserving { _hasListeners = YES; }
- (void)stopObserving  { _hasListeners = NO; }

// ── Backend info ─────────────────────────────────────────────────────────────

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(backendInfo) {
  return @(hilum_backend_info());
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(backendVersion) {
  return @(hilum_backend_version());
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(apiVersion) {
  return @((double)hilum_api_version());
}

// ── Model lifecycle ──────────────────────────────────────────────────────────

RCT_EXPORT_METHOD(loadModel:(NSString *)path
                  options:(NSDictionary *)options
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  dispatch_async(inference_queue(), ^{
    uint64_t available = os_proc_available_memory();
    uint64_t minimumRAM = 512 * 1024 * 1024;
    if (available < minimumRAM) {
      reject(@"E_INSUFFICIENT_MEMORY",
        [NSString stringWithFormat:
          @"Insufficient memory to load model. Available: %llu MB, minimum: %llu MB. "
          @"Close other apps or use a smaller quantized model.",
          available / (1024 * 1024), minimumRAM / (1024 * 1024)],
        nil);
      return;
    }

    hilum_model_params params = hilum_model_default_params();
    if (options[@"n_gpu_layers"]) params.n_gpu_layers = [options[@"n_gpu_layers"] intValue];
    if (options[@"use_mmap"])     params.use_mmap = [options[@"use_mmap"] boolValue];

    hilum_model *model = nullptr;
    hilum_error err = hilum_model_load([path UTF8String], params, &model);
    if (err != HILUM_OK) {
      reject(@"E_MODEL_LOAD", @(hilum_error_str(err)), nil);
      return;
    }

    NSString *modelId = generateUUID();
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      g_models[[modelId UTF8String]] = model;
    }
    resolve(modelId);
  });
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(getModelSize:(NSString *)modelId) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_models.find([modelId UTF8String]);
  if (it == g_models.end()) return @(0);
  return @((double)hilum_model_size(it->second));
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(freeModel:(NSString *)modelId) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_models.find([modelId UTF8String]);
  if (it != g_models.end()) {
    hilum_model_free(it->second);
    g_models.erase(it);
  }
  return nil;
}

// ── Context lifecycle ────────────────────────────────────────────────────────

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(createContext:(NSString *)modelId
                                        options:(NSDictionary *)options) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_models.find([modelId UTF8String]);
  if (it == g_models.end()) return @"";

  hilum_context_params params = hilum_context_default_params();
  if (options[@"n_ctx"])        params.n_ctx = [options[@"n_ctx"] intValue];
  if (options[@"n_batch"])      params.n_batch = [options[@"n_batch"] intValue];
  if (options[@"n_threads"])    params.n_threads = [options[@"n_threads"] intValue];
  if (options[@"n_seq_max"])    params.n_seq_max = [options[@"n_seq_max"] intValue];
  if (options[@"flash_attn_type"]) params.flash_attn = [options[@"flash_attn_type"] intValue];
  if (options[@"type_k"])       params.type_k = [options[@"type_k"] intValue];
  if (options[@"type_v"])       params.type_v = [options[@"type_v"] intValue];
  if (options[@"draft_model_id"]) {
    NSString *draftId = options[@"draft_model_id"];
    auto dit = g_models.find([draftId UTF8String]);
    if (dit != g_models.end()) params.draft_model = dit->second;
  }
  if (options[@"draft_n_max"])  params.draft_n_max = [options[@"draft_n_max"] intValue];

  hilum_context *ctx = nullptr;
  hilum_error err = hilum_context_create(it->second, params, &ctx);
  if (err != HILUM_OK) return @"";

  NSString *ctxId = generateUUID();
  g_contexts[[ctxId UTF8String]] = ctx;
  return ctxId;
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(getContextSize:(NSString *)contextId) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_contexts.find([contextId UTF8String]);
  if (it == g_contexts.end()) return @(0);
  return @((int)hilum_context_size(it->second));
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(freeContext:(NSString *)contextId) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_contexts.find([contextId UTF8String]);
  if (it != g_contexts.end()) {
    hilum_context_free(it->second);
    g_contexts.erase(it);
  }
  return nil;
}

// ── Warmup ───────────────────────────────────────────────────────────────────

RCT_EXPORT_METHOD(warmup:(NSString *)modelId
              contextId:(NSString *)contextId
                resolve:(RCTPromiseResolveBlock)resolve
                 reject:(RCTPromiseRejectBlock)reject) {
  dispatch_async(inference_queue(), ^{
    hilum_model *model;
    hilum_context *ctx;
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      auto mi = g_models.find([modelId UTF8String]);
      auto ci = g_contexts.find([contextId UTF8String]);
      if (mi == g_models.end() || ci == g_contexts.end()) {
        reject(@"E_NOT_FOUND", @"Model or context not found", nil);
        return;
      }
      model = mi->second;
      ctx = ci->second;
    }

    hilum_error err = hilum_warmup(model, ctx);
    if (err != HILUM_OK) {
      reject(@"E_WARMUP", @(hilum_error_str(err)), nil);
      return;
    }
    resolve(nil);
  });
}

// ── Performance metrics ───────────────────────────────────────────────────────

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(getPerf:(NSString *)contextId) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_contexts.find([contextId UTF8String]);
  if (it == g_contexts.end()) return @{};

  hilum_perf_data perf = hilum_get_perf(it->second);
  return @{
    @"promptEvalMs":          @(perf.prompt_eval_ms),
    @"generationMs":          @(perf.generation_ms),
    @"promptTokens":          @(perf.prompt_tokens),
    @"generatedTokens":       @(perf.generated_tokens),
    @"promptTokensPerSec":    @(perf.prompt_tokens_per_sec),
    @"generatedTokensPerSec": @(perf.generated_tokens_per_sec),
  };
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(optimalThreadCount) {
  return @(hilum_optimal_thread_count());
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(benchmark:(NSString *)modelId
                                        contextId:(NSString *)contextId
                                        options:(NSDictionary *)options) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto mi = g_models.find([modelId UTF8String]);
  auto ci = g_contexts.find([contextId UTF8String]);
  if (mi == g_models.end() || ci == g_contexts.end()) return @{};

  hilum_benchmark_params params = hilum_benchmark_default_params();
  if (options[@"promptTokens"])   params.prompt_tokens = [options[@"promptTokens"] intValue];
  if (options[@"generateTokens"]) params.generate_tokens = [options[@"generateTokens"] intValue];
  if (options[@"iterations"])     params.iterations = [options[@"iterations"] intValue];

  hilum_benchmark_result result{};
  hilum_error err = hilum_benchmark(mi->second, ci->second, params, &result);
  if (err != HILUM_OK) {
    return @{ @"error": @(hilum_error_str(err)) };
  }

  return @{
    @"promptTokensPerSec": @(result.prompt_tokens_per_sec),
    @"generatedTokensPerSec": @(result.generated_tokens_per_sec),
    @"ttftMs": @(result.ttft_ms),
    @"totalMs": @(result.total_ms),
    @"iterations": @(result.iterations),
  };
}

// ── KV cache ─────────────────────────────────────────────────────────────────

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(kvCacheClear:(NSString *)contextId
                                        fromPos:(double)fromPos) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_contexts.find([contextId UTF8String]);
  if (it != g_contexts.end()) {
    hilum_context_kv_clear(it->second, (int)fromPos);
  }
  return nil;
}

// ── Tokenization ─────────────────────────────────────────────────────────────

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(tokenize:(NSString *)modelId
                                          text:(NSString *)text
                                    addSpecial:(BOOL)addSpecial
                                  parseSpecial:(BOOL)parseSpecial) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_models.find([modelId UTF8String]);
  if (it == g_models.end()) return @[];

  const char *ctext = [text UTF8String];
  int text_len = (int)strlen(ctext);

  int32_t n = hilum_tokenize(it->second, ctext, text_len, nullptr, 0, addSpecial, parseSpecial);
  if (n >= 0) return @[];

  int32_t n_tokens = -n;
  std::vector<int32_t> tokens(n_tokens);
  n = hilum_tokenize(it->second, ctext, text_len, tokens.data(), n_tokens, addSpecial, parseSpecial);
  if (n < 0) return @[];

  NSMutableArray *result = [NSMutableArray arrayWithCapacity:n];
  for (int i = 0; i < n; i++) {
    [result addObject:@(tokens[i])];
  }
  return result;
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(detokenize:(NSString *)modelId
                                          tokens:(NSArray<NSNumber *> *)tokens) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_models.find([modelId UTF8String]);
  if (it == g_models.end()) return @"";

  std::vector<int32_t> tok_vec;
  tok_vec.reserve(tokens.count);
  for (NSNumber *tok in tokens) tok_vec.push_back([tok intValue]);

  std::vector<char> buf(tok_vec.size() * 16 + 256);
  int32_t n = hilum_detokenize(it->second, tok_vec.data(), (int32_t)tok_vec.size(),
                                buf.data(), (int32_t)buf.size());
  if (n < 0) {
    buf.resize(-n);
    n = hilum_detokenize(it->second, tok_vec.data(), (int32_t)tok_vec.size(),
                          buf.data(), (int32_t)buf.size());
  }
  if (n <= 0) return @"";
  return [[NSString alloc] initWithBytes:buf.data() length:n encoding:NSUTF8StringEncoding] ?: @"";
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(applyChatTemplate:(NSString *)modelId
                                            messages:(NSArray<NSDictionary *> *)messages
                                        addAssistant:(BOOL)addAssistant) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_models.find([modelId UTF8String]);
  if (it == g_models.end()) return @"";

  // Build JSON array
  NSError *jsonError = nil;
  NSData *jsonData = [NSJSONSerialization dataWithJSONObject:messages options:0 error:&jsonError];
  if (jsonError || !jsonData) return @"";

  NSString *jsonStr = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
  const char *cjson = [jsonStr UTF8String];

  std::vector<char> buf(strlen(cjson) * 4 + 256);
  int32_t len = hilum_chat_template(it->second, cjson, addAssistant,
                                     buf.data(), (int32_t)buf.size());
  if (len <= 0) {
    if (len < 0) {
      buf.resize(-len + 1);
      len = hilum_chat_template(it->second, cjson, addAssistant,
                                 buf.data(), (int32_t)buf.size());
    }
    if (len <= 0) return @"";
  }
  return [[NSString alloc] initWithBytes:buf.data() length:len encoding:NSUTF8StringEncoding] ?: @"";
}

// ── Text inference ───────────────────────────────────────────────────────────

RCT_EXPORT_METHOD(generate:(NSString *)modelId
              contextId:(NSString *)contextId
                 prompt:(NSString *)prompt
                options:(NSDictionary *)options
                resolve:(RCTPromiseResolveBlock)resolve
                 reject:(RCTPromiseRejectBlock)reject) {
  dispatch_async(inference_queue(), ^{
    hilum_model *model;
    hilum_context *ctx;
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      auto mi = g_models.find([modelId UTF8String]);
      auto ci = g_contexts.find([contextId UTF8String]);
      if (mi == g_models.end() || ci == g_contexts.end()) {
        reject(@"E_NOT_FOUND", @"Model or context not found", nil);
        return;
      }
      model = mi->second;
      ctx = ci->second;
    }

    GenContext gc = parse_gen_context(options);

    std::vector<char> buf(gc.params.max_tokens * 64 + 1024);
    int32_t generated = 0;

    hilum_error err = hilum_generate(model, ctx, [prompt UTF8String], gc.params,
                                      buf.data(), (int32_t)buf.size(), &generated);
    if (err != HILUM_OK) {
      reject(@"E_GENERATE", @(hilum_error_str(err)), nil);
      return;
    }

    resolve(@(buf.data()));
  });
}

RCT_EXPORT_METHOD(startStream:(NSString *)modelId
               contextId:(NSString *)contextId
                  prompt:(NSString *)prompt
                 options:(NSDictionary *)options) {
  std::string ctxIdStr = [contextId UTF8String];

  dispatch_async(inference_queue(), ^{
    hilum_model *model;
    hilum_context *ctx;
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      auto mi = g_models.find([modelId UTF8String]);
      auto ci = g_contexts.find(ctxIdStr);
      if (mi == g_models.end() || ci == g_contexts.end()) {
        [self sendEventWithName:@"onToken" body:@{
          @"contextId": contextId, @"done": @YES, @"error": @"Model or context not found"
        }];
        return;
      }
      model = mi->second;
      ctx = ci->second;
    }
    hilum_cancel_clear(ctx);

    GenContext gc = parse_gen_context(options);

    struct StreamState {
      __weak LocalLLM *module;
      NSString *contextId;
    };

    StreamState state = { self, contextId };

    hilum_error err = hilum_generate_stream(model, ctx, [prompt UTF8String], gc.params,
      [](const char *token, int32_t token_len, void *ud) -> bool {
        auto *s = static_cast<StreamState *>(ud);
        NSString *piece = [[NSString alloc] initWithBytes:token length:token_len
                                                 encoding:NSUTF8StringEncoding];
        [s->module sendEventWithName:@"onToken" body:@{
          @"contextId": s->contextId,
          @"token": piece ?: @"",
          @"done": @NO,
        }];
        return true;
      }, &state);
    [self sendEventWithName:@"onToken" body:@{
      @"contextId": contextId, @"done": @YES
    }];
  });
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(stopStream:(NSString *)contextId) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_contexts.find([contextId UTF8String]);
  if (it != g_contexts.end()) {
    hilum_cancel(it->second);
  }
  return nil;
}

// ── Vision ───────────────────────────────────────────────────────────────────

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(loadProjector:(NSString *)modelId
                                              path:(NSString *)path
                                           options:(NSDictionary *)options) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_models.find([modelId UTF8String]);
  if (it == g_models.end()) return @"";

  hilum_mtmd_params mparams;
  mparams.use_gpu = options[@"use_gpu"] ? [options[@"use_gpu"] boolValue] : true;
  mparams.n_threads = options[@"n_threads"] ? [options[@"n_threads"] intValue] : 0;

  hilum_mtmd *mtmd = nullptr;
  hilum_error err = hilum_mtmd_load(it->second, [path UTF8String], mparams, &mtmd);
  if (err != HILUM_OK) return @"";

  NSString *mtmdId = generateUUID();
  g_mtmd_contexts[[mtmdId UTF8String]] = mtmd;
  return mtmdId;
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(supportVision:(NSString *)mtmdId) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_mtmd_contexts.find([mtmdId UTF8String]);
  if (it == g_mtmd_contexts.end()) return @NO;
  return @(hilum_mtmd_supports_vision(it->second));
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(freeMtmdContext:(NSString *)mtmdId) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_mtmd_contexts.find([mtmdId UTF8String]);
  if (it != g_mtmd_contexts.end()) {
    hilum_mtmd_free(it->second);
    g_mtmd_contexts.erase(it);
  }
  return nil;
}

RCT_EXPORT_METHOD(generateVision:(NSString *)modelId
                     contextId:(NSString *)contextId
                        mtmdId:(NSString *)mtmdId
                        prompt:(NSString *)prompt
                  imageBase64s:(NSArray<NSString *> *)imageBase64s
                       options:(NSDictionary *)options
                       resolve:(RCTPromiseResolveBlock)resolve
                        reject:(RCTPromiseRejectBlock)reject) {
  dispatch_async(inference_queue(), ^{
    hilum_model *model;
    hilum_context *ctx;
    hilum_mtmd *mctx;
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      auto mi = g_models.find([modelId UTF8String]);
      auto ci = g_contexts.find([contextId UTF8String]);
      auto vi = g_mtmd_contexts.find([mtmdId UTF8String]);
      if (mi == g_models.end() || ci == g_contexts.end() || vi == g_mtmd_contexts.end()) {
        reject(@"E_NOT_FOUND", @"Model, context, or vision context not found", nil);
        return;
      }
      model = mi->second;
      ctx = ci->second;
      mctx = vi->second;
    }

    // Decode base64 images
    std::vector<std::vector<uint8_t>> img_data;
    std::vector<hilum_image> images;
    for (NSString *b64 in imageBase64s) {
      auto data = decode_base64(b64);
      if (data.empty()) continue;
      img_data.push_back(std::move(data));
    }
    for (auto &d : img_data) {
      images.push_back({d.data(), d.size()});
    }

    GenContext gc = parse_gen_context(options);
    std::vector<char> buf(gc.params.max_tokens * 64 + 1024);
    int32_t generated = 0;

    hilum_error err = hilum_generate_vision(model, ctx, mctx, [prompt UTF8String],
      images.data(), (int32_t)images.size(), gc.params,
      buf.data(), (int32_t)buf.size(), &generated);

    if (err != HILUM_OK) {
      reject(@"E_VISION", @(hilum_error_str(err)), nil);
      return;
    }

    resolve(@(buf.data()));
  });
}

RCT_EXPORT_METHOD(startStreamVision:(NSString *)modelId
                        contextId:(NSString *)contextId
                           mtmdId:(NSString *)mtmdId
                           prompt:(NSString *)prompt
                     imageBase64s:(NSArray<NSString *> *)imageBase64s
                          options:(NSDictionary *)options) {
  std::string ctxIdStr = [contextId UTF8String];

  dispatch_async(inference_queue(), ^{
    hilum_model *model;
    hilum_context *ctx;
    hilum_mtmd *mctx;
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      auto mi = g_models.find([modelId UTF8String]);
      auto ci = g_contexts.find(ctxIdStr);
      auto vi = g_mtmd_contexts.find([mtmdId UTF8String]);
      if (mi == g_models.end() || ci == g_contexts.end() || vi == g_mtmd_contexts.end()) {
        [self sendEventWithName:@"onToken" body:@{
          @"contextId": contextId, @"done": @YES, @"error": @"Not found"
        }];
        return;
      }
      model = mi->second;
      ctx = ci->second;
      mctx = vi->second;
    }
    hilum_cancel_clear(ctx);

    std::vector<std::vector<uint8_t>> img_data;
    std::vector<hilum_image> images;
    for (NSString *b64 in imageBase64s) {
      auto data = decode_base64(b64);
      if (data.empty()) continue;
      img_data.push_back(std::move(data));
    }
    for (auto &d : img_data) {
      images.push_back({d.data(), d.size()});
    }

    GenContext gc = parse_gen_context(options);

    struct StreamState {
      __weak LocalLLM *module;
      NSString *contextId;
    };
    StreamState state = { self, contextId };

    hilum_generate_vision_stream(model, ctx, mctx, [prompt UTF8String],
      images.data(), (int32_t)images.size(), gc.params,
      [](const char *token, int32_t token_len, void *ud) -> bool {
        auto *s = static_cast<StreamState *>(ud);
        NSString *piece = [[NSString alloc] initWithBytes:token length:token_len
                                                 encoding:NSUTF8StringEncoding];
        [s->module sendEventWithName:@"onToken" body:@{
          @"contextId": s->contextId,
          @"token": piece ?: @"",
          @"done": @NO,
        }];
        return true;
      }, &state);
    [self sendEventWithName:@"onToken" body:@{
      @"contextId": contextId, @"done": @YES
    }];
  });
}

// ── Grammar ──────────────────────────────────────────────────────────────────

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(jsonSchemaToGrammar:(NSString *)schemaJson) {
  const char *cjson = [schemaJson UTF8String];
  std::vector<char> buf(strlen(cjson) * 8 + 4096);
  int32_t len = hilum_json_schema_to_grammar(cjson, buf.data(), (int32_t)buf.size());
  if (len <= 0) {
    if (len < 0) {
      buf.resize(-len);
      len = hilum_json_schema_to_grammar(cjson, buf.data(), (int32_t)buf.size());
    }
    if (len <= 0) return @"";
  }
  return [[NSString alloc] initWithBytes:buf.data() length:len encoding:NSUTF8StringEncoding] ?: @"";
}

// ── Embeddings ───────────────────────────────────────────────────────────────

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(getEmbeddingDimension:(NSString *)modelId) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_models.find([modelId UTF8String]);
  if (it == g_models.end()) return @(0);
  return @((int)hilum_emb_dimension(it->second));
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(createEmbeddingContext:(NSString *)modelId
                                                      options:(NSDictionary *)options) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_models.find([modelId UTF8String]);
  if (it == g_models.end()) return @"";

  hilum_emb_params params;
  params.n_ctx        = options[@"n_ctx"]        ? [options[@"n_ctx"] intValue]        : 0;
  params.n_batch      = options[@"n_batch"]      ? [options[@"n_batch"] intValue]      : 0;
  params.n_threads    = options[@"n_threads"]    ? [options[@"n_threads"] intValue]    : 0;
  params.pooling_type = options[@"pooling_type"] ? [options[@"pooling_type"] intValue] : -1;

  hilum_emb_ctx *ectx = nullptr;
  hilum_error err = hilum_emb_context_create(it->second, params, &ectx);
  if (err != HILUM_OK) return @"";

  NSString *ctxId = generateUUID();
  g_emb_contexts[[ctxId UTF8String]] = ectx;
  return ctxId;
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(embed:(NSString *)contextId
                                      modelId:(NSString *)modelId
                                       tokens:(NSArray<NSNumber *> *)tokens) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto ci = g_emb_contexts.find([contextId UTF8String]);
  auto mi = g_models.find([modelId UTF8String]);
  if (ci == g_emb_contexts.end() || mi == g_models.end()) return @[];

  int n = (int)tokens.count;
  std::vector<int32_t> tok_vec(n);
  for (int i = 0; i < n; i++) tok_vec[i] = [tokens[i] intValue];

  int n_embd = hilum_emb_dimension(mi->second);
  std::vector<float> emb(n_embd);

  hilum_error err = hilum_embed(ci->second, mi->second, tok_vec.data(), n, emb.data(), n_embd);
  if (err != HILUM_OK) return @[];

  NSMutableArray *result = [NSMutableArray arrayWithCapacity:n_embd];
  for (int i = 0; i < n_embd; i++) [result addObject:@(emb[i])];
  return result;
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(embedBatch:(NSString *)contextId
                                          modelId:(NSString *)modelId
                                     tokenArrays:(NSArray<NSArray<NSNumber *> *> *)tokenArrays) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto ci = g_emb_contexts.find([contextId UTF8String]);
  auto mi = g_models.find([modelId UTF8String]);
  if (ci == g_emb_contexts.end() || mi == g_models.end()) return @[];

  int n_seqs = (int)tokenArrays.count;
  int n_embd = hilum_emb_dimension(mi->second);

  std::vector<std::vector<int32_t>> tok_vecs(n_seqs);
  std::vector<const int32_t *> tok_ptrs(n_seqs);
  std::vector<int32_t> tok_counts(n_seqs);

  for (int s = 0; s < n_seqs; s++) {
    NSArray *toks = tokenArrays[s];
    tok_vecs[s].resize(toks.count);
    for (int i = 0; i < (int)toks.count; i++) tok_vecs[s][i] = [toks[i] intValue];
    tok_ptrs[s] = tok_vecs[s].data();
    tok_counts[s] = (int32_t)tok_vecs[s].size();
  }

  std::vector<std::vector<float>> emb_vecs(n_seqs, std::vector<float>(n_embd));
  std::vector<float *> emb_ptrs(n_seqs);
  for (int s = 0; s < n_seqs; s++) emb_ptrs[s] = emb_vecs[s].data();

  hilum_error err = hilum_embed_batch(ci->second, mi->second,
    tok_ptrs.data(), tok_counts.data(), n_seqs, emb_ptrs.data(), n_embd);
  if (err != HILUM_OK) return @[];

  NSMutableArray *results = [NSMutableArray arrayWithCapacity:n_seqs];
  for (int s = 0; s < n_seqs; s++) {
    NSMutableArray *vec = [NSMutableArray arrayWithCapacity:n_embd];
    for (int i = 0; i < n_embd; i++) [vec addObject:@(emb_vecs[s][i])];
    [results addObject:vec];
  }
  return results;
}

// ── Batch inference ──────────────────────────────────────────────────────────

RCT_EXPORT_METHOD(startBatch:(NSString *)modelId
               contextId:(NSString *)contextId
                 prompts:(NSArray<NSString *> *)prompts
                 options:(NSDictionary *)options) {
  std::string ctxIdStr = [contextId UTF8String];

  dispatch_async(inference_queue(), ^{
    hilum_model *model;
    hilum_context *ctx;
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      auto mi = g_models.find([modelId UTF8String]);
      auto ci = g_contexts.find(ctxIdStr);
      if (mi == g_models.end() || ci == g_contexts.end()) {
        [self sendEventWithName:@"onBatchToken" body:@{
          @"contextId": contextId, @"done": @YES, @"error": @"Not found", @"seqIndex": @(-1)
        }];
        return;
      }
      model = mi->second;
      ctx = ci->second;
    }
    hilum_cancel_clear(ctx);

    int n_seqs = (int)prompts.count;
    std::vector<std::string> prompt_strs(n_seqs);
    std::vector<const char *> prompt_ptrs(n_seqs);
    for (int i = 0; i < n_seqs; i++) {
      prompt_strs[i] = [prompts[i] UTF8String];
      prompt_ptrs[i] = prompt_strs[i].c_str();
    }

    GenContext gc = parse_gen_context(options);

    struct BatchState {
      __weak LocalLLM *module;
      NSString *contextId;
    };
    BatchState state = { self, contextId };

    hilum_generate_batch(model, ctx, prompt_ptrs.data(), n_seqs, gc.params,
      [](hilum_batch_event event, void *ud) -> bool {
        auto *s = static_cast<BatchState *>(ud);
        if (event.done) {
          NSString *reason = event.finish_reason ? @(event.finish_reason) : @"stop";
          [s->module sendEventWithName:@"onBatchToken" body:@{
            @"contextId": s->contextId, @"seqIndex": @(event.seq_index),
            @"done": @YES, @"finishReason": reason
          }];
        } else {
          NSString *piece = [[NSString alloc] initWithBytes:event.token
                                                    length:event.token_len
                                                  encoding:NSUTF8StringEncoding];
          [s->module sendEventWithName:@"onBatchToken" body:@{
            @"contextId": s->contextId, @"seqIndex": @(event.seq_index),
            @"token": piece ?: @"", @"done": @NO
          }];
        }
        return true;
      }, &state);
  });
}

// ── Quantization ─────────────────────────────────────────────────────────────

RCT_EXPORT_METHOD(quantize:(NSString *)inputPath
             outputPath:(NSString *)outputPath
                options:(NSDictionary *)options) {
  dispatch_async(inference_queue(), ^{
    hilum_quantize_params params = hilum_quantize_default_params();
    if (options[@"ftype"])                  params.ftype = [options[@"ftype"] intValue];
    if (options[@"nthread"])                params.nthread = [options[@"nthread"] intValue];
    if (options[@"allow_requantize"])       params.allow_requantize = [options[@"allow_requantize"] boolValue];
    if (options[@"quantize_output_tensor"]) params.quantize_output_tensor = [options[@"quantize_output_tensor"] boolValue];
    if (options[@"pure"])                   params.pure = [options[@"pure"] boolValue];

    hilum_error err = hilum_quantize([inputPath UTF8String], [outputPath UTF8String], params);

    NSString *error = (err != HILUM_OK)
      ? [NSString stringWithFormat:@"Quantization failed: %s", hilum_error_str(err)]
      : nil;
    [self sendEventWithName:@"onQuantizeComplete" body:@{
      @"error": error ?: [NSNull null],
    }];
  });
}

// ── Logging ──────────────────────────────────────────────────────────────────

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(setLogLevel:(double)level) {
  hilum_log_set_level(static_cast<hilum_log_level>((int)level));
  return nil;
}

RCT_EXPORT_METHOD(enableLogEvents:(BOOL)enabled) {
  g_log_events_enabled.store(enabled, std::memory_order_relaxed);
  if (enabled) {
    g_log_module = self;
    hilum_log_set([](hilum_log_level level, const char *text, void *) {
      if (!g_log_events_enabled.load(std::memory_order_relaxed)) return;
      LocalLLM *module = g_log_module;
      if (!module) return;
      [module sendEventWithName:@"onLog" body:@{
        @"level": @((int)level),
        @"text": @(text),
      }];
    }, nullptr);
  } else {
    hilum_log_set(nullptr, nullptr);
    g_log_module = nil;
  }
}

// ── Downloads ────────────────────────────────────────────────────────────────

RCT_EXPORT_METHOD(downloadModel:(NSString *)url destPath:(NSString *)destPath) {
  NSURL *nsUrl = [NSURL URLWithString:url];
  if (!nsUrl) return;
  _downloadDelegate.destPaths[url] = destPath;
  NSURLSessionDownloadTask *task = [_downloadSession downloadTaskWithURL:nsUrl];
  [task resume];
}

RCT_EXPORT_METHOD(resumeDownload:(NSString *)url destPath:(NSString *)destPath) {
  NSData *data = _downloadDelegate.resumeData[url];
  _downloadDelegate.destPaths[url] = destPath;
  if (data) {
    [_downloadDelegate.resumeData removeObjectForKey:url];
    NSURLSessionDownloadTask *task = [_downloadSession downloadTaskWithResumeData:data];
    [task resume];
  } else {
    // No resume data available — fall back to a fresh download
    NSURL *nsUrl = [NSURL URLWithString:url];
    if (!nsUrl) return;
    NSURLSessionDownloadTask *task = [_downloadSession downloadTaskWithURL:nsUrl];
    [task resume];
  }
}

RCT_EXPORT_METHOD(cancelDownload:(NSString *)url) {
  [_downloadSession getTasksWithCompletionHandler:^(NSArray *dataTasks, NSArray *uploadTasks, NSArray *downloadTasks) {
    for (NSURLSessionDownloadTask *task in downloadTasks) {
      if ([task.originalRequest.URL.absoluteString isEqualToString:url]) {
        [task cancelByProducingResumeData:^(NSData *resumeData) {
          if (resumeData) {
            self->_downloadDelegate.resumeData[url] = resumeData;
          }
        }];
      }
    }
  }];
}

// ── Device capabilities ──────────────────────────────────────────────────────

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(getDeviceCapabilities) {
  NSProcessInfo *info = [NSProcessInfo processInfo];
  id<MTLDevice> gpu = MTLCreateSystemDefaultDevice();

  uint64_t totalRAM = info.physicalMemory;
  uint64_t availableRAM = os_proc_available_memory();

  NSOperatingSystemVersion ver = info.operatingSystemVersion;
  NSString *iosVersion = [NSString stringWithFormat:@"%ld.%ld.%ld",
    (long)ver.majorVersion, (long)ver.minorVersion, (long)ver.patchVersion];

  int metalFamily = 0;
  if (gpu) {
    if ([gpu supportsFamily:MTLGPUFamilyApple9]) metalFamily = 9;
    else if ([gpu supportsFamily:MTLGPUFamilyApple8]) metalFamily = 8;
    else if ([gpu supportsFamily:MTLGPUFamilyApple7]) metalFamily = 7;
    else if ([gpu supportsFamily:MTLGPUFamilyApple6]) metalFamily = 6;
    else if ([gpu supportsFamily:MTLGPUFamilyApple5]) metalFamily = 5;
    else if ([gpu supportsFamily:MTLGPUFamilyApple4]) metalFamily = 4;
  }

  int metalVersion = metalFamily >= 7 ? 3 : metalFamily >= 5 ? 2 : 1;

  return @{
    @"totalRAM":        @(totalRAM),
    @"availableRAM":    @(availableRAM),
    @"gpuName":         gpu ? gpu.name : @"unknown",
    @"metalFamily":     @(metalFamily),
    @"metalVersion":    @(metalVersion),
    @"iosVersion":      iosVersion,
    @"isLowPowerMode":  @(info.isLowPowerModeEnabled),
  };
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(getModelStoragePath) {
  NSArray *paths = NSSearchPathForDirectoriesInDomains(NSApplicationSupportDirectory, NSUserDomainMask, YES);
  NSString *appSupport = paths.firstObject;
  NSString *llmDir = [appSupport stringByAppendingPathComponent:@"local-llm/models"];
  NSFileManager *fm = [NSFileManager defaultManager];
  if (![fm fileExistsAtPath:llmDir]) {
    [fm createDirectoryAtPath:llmDir withIntermediateDirectories:YES attributes:nil error:nil];
  }
  return llmDir;
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(fileExists:(NSString *)path) {
  return @([[NSFileManager defaultManager] fileExistsAtPath:path]);
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(getFileSize:(NSString *)path) {
  NSDictionary *attrs = [[NSFileManager defaultManager] attributesOfItemAtPath:path error:nil];
  NSNumber *size = attrs[NSFileSize];
  return size ?: @0;
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(readTextFile:(NSString *)path) {
  if (!isPathAllowed(path)) return nil;
  NSError *error = nil;
  NSString *content = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
  if (error || !content) {
    return nil;
  }
  return content;
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(writeTextFile:(NSString *)path content:(NSString *)content) {
  if (!isPathAllowed(path)) return @(NO);
  NSFileManager *fm = [NSFileManager defaultManager];
  NSString *dir = [path stringByDeletingLastPathComponent];
  if (![fm fileExistsAtPath:dir]) {
    [fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
  }
  NSError *error = nil;
  [content writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:&error];
  return @(error == nil);
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(removePath:(NSString *)path) {
  if (!isPathAllowed(path)) return @(NO);
  NSError *error = nil;
  [[NSFileManager defaultManager] removeItemAtPath:path error:&error];
  return @(error == nil);
}

RCT_EXPORT_METHOD(sha256File:(NSString *)path
                      resolve:(RCTPromiseResolveBlock)resolve
                       reject:(RCTPromiseRejectBlock)reject) {
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    NSInputStream *stream = [NSInputStream inputStreamWithFileAtPath:path];
    if (!stream) {
      reject(@"E_FILE_NOT_FOUND", @"File not found", nil);
      return;
    }

    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);

    [stream open];
    uint8_t buffer[65536];
    while ([stream hasBytesAvailable]) {
      NSInteger bytesRead = [stream read:buffer maxLength:sizeof(buffer)];
      if (bytesRead > 0) {
        CC_SHA256_Update(&ctx, buffer, (CC_LONG)bytesRead);
      }
    }
    [stream close];

    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &ctx);

    NSMutableString *hex = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
      [hex appendFormat:@"%02x", digest[i]];
    }
    resolve(hex);
  });
}

@end
