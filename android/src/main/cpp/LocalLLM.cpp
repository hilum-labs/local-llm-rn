/*
 * LocalLLM.cpp — JNI bridge for local-llm-rn (Android).
 *
 * Mirrors ios/LocalLLM.mm exactly: same handle maps, same hilum_llm.h C API
 * calls, same stream cancellation pattern. The only differences are JNI
 * marshalling instead of Obj-C types, and Android-specific device capabilities.
 */

#include <jni.h>
#include <android/log.h>

#include <string>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <chrono>

#include "hilum_llm.h"
#include "ggml.h"
#include "ggml-backend.h"

#define TAG "LocalLLM"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ── UUID generation ──────────────────────────────────────────────────────────

static std::string generate_uuid() {
    static std::atomic<uint64_t> counter{0};
    // Simple pseudo-UUID: timestamp + counter
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    uint64_t c = counter.fetch_add(1);
    char buf[64];
    snprintf(buf, sizeof(buf), "%llx-%llx",
             (unsigned long long)ms, (unsigned long long)c);
    return buf;
}

// ── Handle maps ──────────────────────────────────────────────────────────────

static std::mutex g_mutex;

static std::unordered_map<std::string, hilum_model *> g_models;
static std::unordered_map<std::string, hilum_context *> g_contexts;
static std::unordered_map<std::string, hilum_mtmd *> g_mtmd_contexts;
static std::unordered_map<std::string, hilum_emb_ctx *> g_emb_contexts;

// ── Log state ────────────────────────────────────────────────────────────────

static std::atomic<bool> g_log_events_enabled{false};
static JavaVM *g_jvm = nullptr;
static jobject g_module_ref = nullptr; // weak global ref
static std::mutex g_log_mutex;

// ── JNI helpers ──────────────────────────────────────────────────────────────

static std::string jstring_to_std(JNIEnv *env, jstring jstr) {
    if (!jstr) return "";
    const char *chars = env->GetStringUTFChars(jstr, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}

static jstring std_to_jstring(JNIEnv *env, const std::string &str) {
    return env->NewStringUTF(str.c_str());
}

static std::string get_string_from_map(JNIEnv *env, jobject map, const char *key) {
    jclass mapClass = env->GetObjectClass(map);
    jmethodID getId = env->GetMethodID(mapClass, "get",
        "(Ljava/lang/Object;)Ljava/lang/Object;");
    jstring jkey = env->NewStringUTF(key);
    jobject value = env->CallObjectMethod(map, getId, jkey);
    env->DeleteLocalRef(jkey);
    env->DeleteLocalRef(mapClass);
    if (!value) return "";
    jclass strClass = env->FindClass("java/lang/String");
    if (env->IsInstanceOf(value, strClass)) {
        std::string result = jstring_to_std(env, (jstring)value);
        env->DeleteLocalRef(strClass);
        env->DeleteLocalRef(value);
        return result;
    }
    env->DeleteLocalRef(strClass);
    env->DeleteLocalRef(value);
    return "";
}

static bool has_key(JNIEnv *env, jobject map, const char *key) {
    jclass mapClass = env->GetObjectClass(map);
    jmethodID containsKey = env->GetMethodID(mapClass, "containsKey",
        "(Ljava/lang/Object;)Z");
    jstring jkey = env->NewStringUTF(key);
    jboolean result = env->CallBooleanMethod(map, containsKey, jkey);
    env->DeleteLocalRef(jkey);
    env->DeleteLocalRef(mapClass);
    return result;
}

static int get_int_from_map(JNIEnv *env, jobject map, const char *key, int defaultVal) {
    if (!has_key(env, map, key)) return defaultVal;
    jclass mapClass = env->GetObjectClass(map);
    jmethodID getId = env->GetMethodID(mapClass, "get",
        "(Ljava/lang/Object;)Ljava/lang/Object;");
    jstring jkey = env->NewStringUTF(key);
    jobject value = env->CallObjectMethod(map, getId, jkey);
    env->DeleteLocalRef(jkey);
    env->DeleteLocalRef(mapClass);
    if (!value) return defaultVal;
    jclass numClass = env->FindClass("java/lang/Number");
    if (env->IsInstanceOf(value, numClass)) {
        jmethodID intValue = env->GetMethodID(numClass, "intValue", "()I");
        int result = env->CallIntMethod(value, intValue);
        env->DeleteLocalRef(numClass);
        env->DeleteLocalRef(value);
        return result;
    }
    env->DeleteLocalRef(numClass);
    env->DeleteLocalRef(value);
    return defaultVal;
}

static float get_float_from_map(JNIEnv *env, jobject map, const char *key, float defaultVal) {
    if (!has_key(env, map, key)) return defaultVal;
    jclass mapClass = env->GetObjectClass(map);
    jmethodID getId = env->GetMethodID(mapClass, "get",
        "(Ljava/lang/Object;)Ljava/lang/Object;");
    jstring jkey = env->NewStringUTF(key);
    jobject value = env->CallObjectMethod(map, getId, jkey);
    env->DeleteLocalRef(jkey);
    env->DeleteLocalRef(mapClass);
    if (!value) return defaultVal;
    jclass numClass = env->FindClass("java/lang/Number");
    if (env->IsInstanceOf(value, numClass)) {
        jmethodID floatValue = env->GetMethodID(numClass, "floatValue", "()F");
        float result = env->CallFloatMethod(value, floatValue);
        env->DeleteLocalRef(numClass);
        env->DeleteLocalRef(value);
        return result;
    }
    env->DeleteLocalRef(numClass);
    env->DeleteLocalRef(value);
    return defaultVal;
}

static bool get_bool_from_map(JNIEnv *env, jobject map, const char *key, bool defaultVal) {
    if (!has_key(env, map, key)) return defaultVal;
    jclass mapClass = env->GetObjectClass(map);
    jmethodID getId = env->GetMethodID(mapClass, "get",
        "(Ljava/lang/Object;)Ljava/lang/Object;");
    jstring jkey = env->NewStringUTF(key);
    jobject value = env->CallObjectMethod(map, getId, jkey);
    env->DeleteLocalRef(jkey);
    env->DeleteLocalRef(mapClass);
    if (!value) return defaultVal;
    jclass boolClass = env->FindClass("java/lang/Boolean");
    if (env->IsInstanceOf(value, boolClass)) {
        jmethodID boolValue = env->GetMethodID(boolClass, "booleanValue", "()Z");
        bool result = env->CallBooleanMethod(value, boolValue);
        env->DeleteLocalRef(boolClass);
        env->DeleteLocalRef(value);
        return result;
    }
    env->DeleteLocalRef(boolClass);
    env->DeleteLocalRef(value);
    return defaultVal;
}

// ── ReadableArray helpers ────────────────────────────────────────────────────

static std::vector<int32_t> jarray_to_int_vec(JNIEnv *env, jobject array) {
    std::vector<int32_t> result;
    if (!array) return result;

    jclass listClass = env->FindClass("java/util/List");
    jmethodID sizeMethod = env->GetMethodID(listClass, "size", "()I");
    jmethodID getMethod = env->GetMethodID(listClass, "get", "(I)Ljava/lang/Object;");

    int size = env->CallIntMethod(array, sizeMethod);
    result.reserve(size);

    jclass numClass = env->FindClass("java/lang/Number");
    jmethodID intValue = env->GetMethodID(numClass, "intValue", "()I");

    for (int i = 0; i < size; i++) {
        jobject elem = env->CallObjectMethod(array, getMethod, i);
        if (elem && env->IsInstanceOf(elem, numClass)) {
            result.push_back(env->CallIntMethod(elem, intValue));
        }
        if (elem) env->DeleteLocalRef(elem);
    }

    env->DeleteLocalRef(listClass);
    env->DeleteLocalRef(numClass);
    return result;
}

// ── Gen params helper (mirrors iOS parse_gen_params) ─────────────────────────

struct GenContext {
    hilum_gen_params params;
    std::string grammar;
    std::string grammar_root;

    void finalize() {
        params.grammar      = grammar.empty()      ? nullptr : grammar.c_str();
        params.grammar_root  = grammar_root.empty() ? nullptr : grammar_root.c_str();
    }
};

static GenContext parse_gen_context(JNIEnv *env, jobject options) {
    GenContext gc;
    gc.params = hilum_gen_default_params();

    if (!options) return gc;

    gc.params.max_tokens       = get_int_from_map(env, options, "max_tokens", gc.params.max_tokens);
    gc.params.temperature      = get_float_from_map(env, options, "temperature", gc.params.temperature);
    gc.params.top_p            = get_float_from_map(env, options, "top_p", gc.params.top_p);
    gc.params.top_k            = get_int_from_map(env, options, "top_k", gc.params.top_k);
    gc.params.repeat_penalty   = get_float_from_map(env, options, "repeat_penalty", gc.params.repeat_penalty);
    gc.params.frequency_penalty = get_float_from_map(env, options, "frequency_penalty", gc.params.frequency_penalty);
    gc.params.presence_penalty = get_float_from_map(env, options, "presence_penalty", gc.params.presence_penalty);
    gc.params.seed             = (uint32_t)get_int_from_map(env, options, "seed", (int)gc.params.seed);
    gc.params.n_past           = get_int_from_map(env, options, "n_past", gc.params.n_past);

    std::string gram = get_string_from_map(env, options, "grammar");
    if (!gram.empty()) gc.grammar = gram;
    std::string gram_root = get_string_from_map(env, options, "grammar_root");
    if (!gram_root.empty()) gc.grammar_root = gram_root;

    gc.finalize();
    return gc;
}

// ── Base64 decoding ──────────────────────────────────────────────────────────

static std::vector<uint8_t> decode_base64(JNIEnv *env, jstring jb64) {
    std::string b64 = jstring_to_std(env, jb64);
    if (b64.empty()) return {};

    jclass base64Class = env->FindClass("android/util/Base64");
    jmethodID decodeMethod = env->GetStaticMethodID(base64Class, "decode",
        "(Ljava/lang/String;I)[B");
    jbyteArray decoded = (jbyteArray)env->CallStaticObjectMethod(
        base64Class, decodeMethod, jb64, 0 /* DEFAULT */);
    env->DeleteLocalRef(base64Class);

    if (!decoded) return {};

    jsize len = env->GetArrayLength(decoded);
    std::vector<uint8_t> result(len);
    env->GetByteArrayRegion(decoded, 0, len, reinterpret_cast<jbyte *>(result.data()));
    env->DeleteLocalRef(decoded);
    return result;
}

// ── JNI callback to Kotlin ───────────────────────────────────────────────────

static void call_kotlin_method(JNIEnv *env, jobject module, const char *method,
                                const char *sig, ...) {
    jclass cls = env->GetObjectClass(module);
    jmethodID mid = env->GetMethodID(cls, method, sig);
    env->DeleteLocalRef(cls);
    if (!mid) return;

    va_list args;
    va_start(args, sig);
    env->CallVoidMethodV(module, mid, args);
    va_end(args);
}

// ── JNI_OnLoad ───────────────────────────────────────────────────────────────

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void * /*reserved*/) {
    g_jvm = vm;
    return JNI_VERSION_1_6;
}

// ── JNI exports ──────────────────────────────────────────────────────────────

#define JNI_FN(name) Java_com_hilum_locallm_LocalLLMModule_##name

extern "C" {

// ── Init (load CPU variant .so files) ────────────────────────────────────────

JNIEXPORT void JNICALL
JNI_FN(nativeInit)(JNIEnv *env, jobject thiz, jstring nativeLibDir) {
    std::string libDir = jstring_to_std(env, nativeLibDir);
    LOGI("Loading backends from: %s", libDir.c_str());
    ggml_backend_load_all_from_path(libDir.c_str());
}

// ── Backend info ─────────────────────────────────────────────────────────────

JNIEXPORT jstring JNICALL
JNI_FN(nativeBackendInfo)(JNIEnv *env, jobject thiz) {
    return std_to_jstring(env, hilum_backend_info());
}

JNIEXPORT jstring JNICALL
JNI_FN(nativeBackendVersion)(JNIEnv *env, jobject thiz) {
    return std_to_jstring(env, hilum_backend_version());
}

JNIEXPORT jint JNICALL
JNI_FN(nativeApiVersion)(JNIEnv *env, jobject thiz) {
    return static_cast<jint>(hilum_api_version());
}

// ── Model lifecycle ──────────────────────────────────────────────────────────

JNIEXPORT jstring JNICALL
JNI_FN(nativeLoadModel)(JNIEnv *env, jobject thiz, jstring path, jobject options) {
    std::string pathStr = jstring_to_std(env, path);

    hilum_model_params params = hilum_model_default_params();
    if (options) {
        params.n_gpu_layers = get_int_from_map(env, options, "n_gpu_layers", params.n_gpu_layers);
        params.use_mmap = get_bool_from_map(env, options, "use_mmap", params.use_mmap);
    }

    hilum_model *model = nullptr;
    hilum_error err = hilum_model_load(pathStr.c_str(), params, &model);
    if (err != HILUM_OK) {
        LOGE("Model load failed: %s", hilum_error_str(err));
        return std_to_jstring(env, "");
    }

    std::string modelId = generate_uuid();
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_models[modelId] = model;
    }
    LOGI("Model loaded: %s", modelId.c_str());
    return std_to_jstring(env, modelId);
}

JNIEXPORT jdouble JNICALL
JNI_FN(nativeGetModelSize)(JNIEnv *env, jobject thiz, jstring modelId) {
    std::string id = jstring_to_std(env, modelId);
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_models.find(id);
    if (it == g_models.end()) return 0.0;
    return (jdouble)hilum_model_size(it->second);
}

JNIEXPORT void JNICALL
JNI_FN(nativeFreeModel)(JNIEnv *env, jobject thiz, jstring modelId) {
    std::string id = jstring_to_std(env, modelId);
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_models.find(id);
    if (it != g_models.end()) {
        hilum_model_free(it->second);
        g_models.erase(it);
    }
}

// ── Context lifecycle ────────────────────────────────────────────────────────

JNIEXPORT jstring JNICALL
JNI_FN(nativeCreateContext)(JNIEnv *env, jobject thiz, jstring modelId, jobject options) {
    std::string mid = jstring_to_std(env, modelId);
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_models.find(mid);
    if (it == g_models.end()) return std_to_jstring(env, "");

    hilum_context_params params = hilum_context_default_params();
    if (options) {
        params.n_ctx        = (uint32_t)get_int_from_map(env, options, "n_ctx", params.n_ctx);
        params.n_batch      = (uint32_t)get_int_from_map(env, options, "n_batch", params.n_batch);
        params.n_threads    = (uint32_t)get_int_from_map(env, options, "n_threads", params.n_threads);
        params.n_seq_max    = (uint32_t)get_int_from_map(env, options, "n_seq_max", params.n_seq_max);
        params.flash_attn   = get_int_from_map(env, options, "flash_attn_type", params.flash_attn);
        params.type_k       = get_int_from_map(env, options, "type_k", params.type_k);
        params.type_v       = get_int_from_map(env, options, "type_v", params.type_v);
        params.draft_n_max  = get_int_from_map(env, options, "draft_n_max", params.draft_n_max);
        std::string draftId = get_string_from_map(env, options, "draft_model_id");
        if (!draftId.empty()) {
            auto dit = g_models.find(draftId);
            if (dit != g_models.end()) params.draft_model = dit->second;
        }
    }

    hilum_context *ctx = nullptr;
    hilum_error err = hilum_context_create(it->second, params, &ctx);
    if (err != HILUM_OK) return std_to_jstring(env, "");

    std::string ctxId = generate_uuid();
    g_contexts[ctxId] = ctx;
    return std_to_jstring(env, ctxId);
}

JNIEXPORT jint JNICALL
JNI_FN(nativeGetContextSize)(JNIEnv *env, jobject thiz, jstring contextId) {
    std::string id = jstring_to_std(env, contextId);
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_contexts.find(id);
    if (it == g_contexts.end()) return 0;
    return (jint)hilum_context_size(it->second);
}

JNIEXPORT void JNICALL
JNI_FN(nativeFreeContext)(JNIEnv *env, jobject thiz, jstring contextId) {
    std::string id = jstring_to_std(env, contextId);
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_contexts.find(id);
    if (it != g_contexts.end()) {
        hilum_context_free(it->second);
        g_contexts.erase(it);
    }
}

// ── Warmup ───────────────────────────────────────────────────────────────────

JNIEXPORT void JNICALL
JNI_FN(nativeWarmup)(JNIEnv *env, jobject thiz, jstring modelId, jstring contextId) {
    std::string mid = jstring_to_std(env, modelId);
    std::string cid = jstring_to_std(env, contextId);

    hilum_model *model;
    hilum_context *ctx;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto mi = g_models.find(mid);
        auto ci = g_contexts.find(cid);
        if (mi == g_models.end() || ci == g_contexts.end()) {
            jclass ex = env->FindClass("java/lang/RuntimeException");
            env->ThrowNew(ex, "Model or context not found");
            return;
        }
        model = mi->second;
        ctx = ci->second;
    }

    hilum_error err = hilum_warmup(model, ctx);
    if (err != HILUM_OK) {
        jclass ex = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(ex, hilum_error_str(err));
    }
}

// ── KV cache ─────────────────────────────────────────────────────────────────

JNIEXPORT void JNICALL
JNI_FN(nativeKvCacheClear)(JNIEnv *env, jobject thiz, jstring contextId, jint fromPos) {
    std::string id = jstring_to_std(env, contextId);
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_contexts.find(id);
    if (it != g_contexts.end()) {
        hilum_context_kv_clear(it->second, (int32_t)fromPos);
    }
}

// ── Tokenization ─────────────────────────────────────────────────────────────

JNIEXPORT jobject JNICALL
JNI_FN(nativeTokenize)(JNIEnv *env, jobject thiz, jstring modelId, jstring text,
                        jboolean addSpecial, jboolean parseSpecial) {
    std::string mid = jstring_to_std(env, modelId);
    std::string txt = jstring_to_std(env, text);

    jclass arrayListClass = env->FindClass("java/util/ArrayList");
    jmethodID initMethod = env->GetMethodID(arrayListClass, "<init>", "()V");
    jmethodID addMethod = env->GetMethodID(arrayListClass, "add", "(Ljava/lang/Object;)Z");
    jobject result = env->NewObject(arrayListClass, initMethod);

    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_models.find(mid);
    if (it == g_models.end()) {
        env->DeleteLocalRef(arrayListClass);
        return result;
    }

    int32_t n = hilum_tokenize(it->second, txt.c_str(), (int32_t)txt.size(),
                                nullptr, 0, addSpecial, parseSpecial);
    if (n >= 0) {
        env->DeleteLocalRef(arrayListClass);
        return result;
    }

    int32_t n_tokens = -n;
    std::vector<int32_t> tokens(n_tokens);
    n = hilum_tokenize(it->second, txt.c_str(), (int32_t)txt.size(),
                        tokens.data(), n_tokens, addSpecial, parseSpecial);
    if (n < 0) {
        env->DeleteLocalRef(arrayListClass);
        return result;
    }

    jclass intClass = env->FindClass("java/lang/Integer");
    jmethodID valueOf = env->GetStaticMethodID(intClass, "valueOf", "(I)Ljava/lang/Integer;");
    for (int i = 0; i < n; i++) {
        jobject intObj = env->CallStaticObjectMethod(intClass, valueOf, tokens[i]);
        env->CallBooleanMethod(result, addMethod, intObj);
        env->DeleteLocalRef(intObj);
    }
    env->DeleteLocalRef(intClass);
    env->DeleteLocalRef(arrayListClass);
    return result;
}

JNIEXPORT jstring JNICALL
JNI_FN(nativeDetokenize)(JNIEnv *env, jobject thiz, jstring modelId, jobject tokens) {
    std::string mid = jstring_to_std(env, modelId);
    std::vector<int32_t> tok_vec = jarray_to_int_vec(env, tokens);

    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_models.find(mid);
    if (it == g_models.end()) return std_to_jstring(env, "");

    std::vector<char> buf(tok_vec.size() * 16 + 256);
    int32_t n = hilum_detokenize(it->second, tok_vec.data(), (int32_t)tok_vec.size(),
                                  buf.data(), (int32_t)buf.size());
    if (n < 0) {
        buf.resize(-n);
        n = hilum_detokenize(it->second, tok_vec.data(), (int32_t)tok_vec.size(),
                              buf.data(), (int32_t)buf.size());
    }
    if (n <= 0) return std_to_jstring(env, "");
    return env->NewStringUTF(std::string(buf.data(), n).c_str());
}

JNIEXPORT jstring JNICALL
JNI_FN(nativeApplyChatTemplate)(JNIEnv *env, jobject thiz, jstring modelId,
                                 jstring messagesJson, jboolean addAssistant) {
    std::string mid = jstring_to_std(env, modelId);
    std::string json = jstring_to_std(env, messagesJson);

    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_models.find(mid);
    if (it == g_models.end()) return std_to_jstring(env, "");

    std::vector<char> buf(json.size() * 4 + 256);
    int32_t len = hilum_chat_template(it->second, json.c_str(), addAssistant,
                                       buf.data(), (int32_t)buf.size());
    if (len <= 0) {
        if (len < 0) {
            buf.resize(-len + 1);
            len = hilum_chat_template(it->second, json.c_str(), addAssistant,
                                       buf.data(), (int32_t)buf.size());
        }
        if (len <= 0) return std_to_jstring(env, "");
    }
    return env->NewStringUTF(std::string(buf.data(), len).c_str());
}

// ── Text inference ───────────────────────────────────────────────────────────

JNIEXPORT jstring JNICALL
JNI_FN(nativeGenerate)(JNIEnv *env, jobject thiz, jstring modelId, jstring contextId,
                        jstring prompt, jobject options) {
    std::string mid = jstring_to_std(env, modelId);
    std::string cid = jstring_to_std(env, contextId);
    std::string promptStr = jstring_to_std(env, prompt);

    hilum_model *model;
    hilum_context *ctx;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto mi = g_models.find(mid);
        auto ci = g_contexts.find(cid);
        if (mi == g_models.end() || ci == g_contexts.end()) {
            return std_to_jstring(env, "");
        }
        model = mi->second;
        ctx = ci->second;
    }

    GenContext gc = parse_gen_context(env, options);

    std::vector<char> buf(gc.params.max_tokens * 64 + 1024);
    int32_t generated = 0;

    hilum_error err = hilum_generate(model, ctx, promptStr.c_str(), gc.params,
                                      buf.data(), (int32_t)buf.size(), &generated);
    if (err != HILUM_OK) {
        LOGE("Generate failed: %s", hilum_error_str(err));
        return std_to_jstring(env, "");
    }

    return env->NewStringUTF(buf.data());
}

JNIEXPORT void JNICALL
JNI_FN(nativeStartStream)(JNIEnv *env, jobject thiz, jstring modelId, jstring contextId,
                           jstring prompt, jobject options) {
    std::string mid = jstring_to_std(env, modelId);
    std::string cid = jstring_to_std(env, contextId);
    std::string promptStr = jstring_to_std(env, prompt);

    hilum_model *model;
    hilum_context *ctx;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto mi = g_models.find(mid);
        auto ci = g_contexts.find(cid);
        if (mi == g_models.end() || ci == g_contexts.end()) {
            // Emit error
            jclass cls = env->GetObjectClass(thiz);
            jmethodID emitMethod = env->GetMethodID(cls, "emitToken",
                "(Ljava/lang/String;Ljava/lang/String;ZLjava/lang/String;)V");
            jstring jcid = std_to_jstring(env, cid);
            jstring jerr = env->NewStringUTF("Model or context not found");
            env->CallVoidMethod(thiz, emitMethod, jcid, nullptr, JNI_TRUE, jerr);
            env->DeleteLocalRef(jcid);
            env->DeleteLocalRef(jerr);
            env->DeleteLocalRef(cls);
            return;
        }
        model = mi->second;
        ctx = ci->second;
    }
    hilum_cancel_clear(ctx);

    GenContext gc = parse_gen_context(env, options);

    // Store module ref for callback
    jobject moduleRef = env->NewGlobalRef(thiz);

    struct StreamState {
        jobject moduleRef;
        std::string ctxId;
        JavaVM *jvm;
    };

    StreamState *state = new StreamState{moduleRef, cid, g_jvm};

    hilum_error err = hilum_generate_stream(model, ctx, promptStr.c_str(), gc.params,
        [](const char *token, int32_t token_len, void *ud) -> bool {
            auto *s = static_cast<StreamState *>(ud);

            JNIEnv *env;
            bool detach = false;
            if (s->jvm->GetEnv((void **)&env, JNI_VERSION_1_6) == JNI_EDETACHED) {
                s->jvm->AttachCurrentThread(&env, nullptr);
                detach = true;
            }

            jclass cls = env->GetObjectClass(s->moduleRef);
            jmethodID emitMethod = env->GetMethodID(cls, "emitToken",
                "(Ljava/lang/String;Ljava/lang/String;ZLjava/lang/String;)V");

            jstring jcid = env->NewStringUTF(s->ctxId.c_str());
            jstring jtok = env->NewStringUTF(std::string(token, token_len).c_str());
            env->CallVoidMethod(s->moduleRef, emitMethod, jcid, jtok, JNI_FALSE, nullptr);
            env->DeleteLocalRef(jcid);
            env->DeleteLocalRef(jtok);
            env->DeleteLocalRef(cls);

            if (detach) s->jvm->DetachCurrentThread();
            return true;
        }, state);

    // Emit done
    {
        jclass cls = env->GetObjectClass(thiz);
        jmethodID emitMethod = env->GetMethodID(cls, "emitToken",
            "(Ljava/lang/String;Ljava/lang/String;ZLjava/lang/String;)V");
        jstring jcid = std_to_jstring(env, cid);
        env->CallVoidMethod(thiz, emitMethod, jcid, nullptr, JNI_TRUE, nullptr);
        env->DeleteLocalRef(jcid);
        env->DeleteLocalRef(cls);
    }

    env->DeleteGlobalRef(moduleRef);
    delete state;
}

JNIEXPORT void JNICALL
JNI_FN(nativeStopStream)(JNIEnv *env, jobject thiz, jstring contextId) {
    std::string cid = jstring_to_std(env, contextId);
    std::lock_guard<std::mutex> lock(g_mutex);
    auto ci = g_contexts.find(cid);
    if (ci != g_contexts.end()) {
        hilum_cancel(ci->second);
    }
}

JNIEXPORT jobject JNICALL
JNI_FN(nativeGetPerf)(JNIEnv *env, jobject thiz, jstring contextId) {
    std::string cid = jstring_to_std(env, contextId);

    jclass mapClass = env->FindClass("java/util/HashMap");
    jmethodID mapInit = env->GetMethodID(mapClass, "<init>", "()V");
    jmethodID mapPut = env->GetMethodID(mapClass, "put",
        "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jobject map = env->NewObject(mapClass, mapInit);

    jclass doubleClass = env->FindClass("java/lang/Double");
    jmethodID doubleInit = env->GetMethodID(doubleClass, "<init>", "(D)V");

    hilum_context *ctx = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto ci = g_contexts.find(cid);
        if (ci == g_contexts.end()) return map;
        ctx = ci->second;
    }

    hilum_perf_data perf = hilum_get_perf(ctx);

    auto putDouble = [&](const char *key, double val) {
        jstring jkey = env->NewStringUTF(key);
        jobject jval = env->NewObject(doubleClass, doubleInit, val);
        env->CallObjectMethod(map, mapPut, jkey, jval);
        env->DeleteLocalRef(jkey);
        env->DeleteLocalRef(jval);
    };

    putDouble("promptEvalMs",          perf.prompt_eval_ms);
    putDouble("generationMs",          perf.generation_ms);
    putDouble("promptTokens",          (double)perf.prompt_tokens);
    putDouble("generatedTokens",       (double)perf.generated_tokens);
    putDouble("promptTokensPerSec",    perf.prompt_tokens_per_sec);
    putDouble("generatedTokensPerSec", perf.generated_tokens_per_sec);

    env->DeleteLocalRef(mapClass);
    env->DeleteLocalRef(doubleClass);
    return map;
}

JNIEXPORT jint JNICALL
JNI_FN(nativeOptimalThreadCount)(JNIEnv *env, jobject thiz) {
    return (jint)hilum_optimal_thread_count();
}

JNIEXPORT jobject JNICALL
JNI_FN(nativeBenchmark)(JNIEnv *env, jobject thiz, jstring modelId, jstring contextId,
                         jobject options) {
    std::string mid = jstring_to_std(env, modelId);
    std::string cid = jstring_to_std(env, contextId);

    jclass mapClass = env->FindClass("java/util/HashMap");
    jmethodID mapInit = env->GetMethodID(mapClass, "<init>", "()V");
    jmethodID mapPut = env->GetMethodID(mapClass, "put",
        "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jobject map = env->NewObject(mapClass, mapInit);

    auto putDouble = [&](const char *key, double val) {
        jclass doubleClass = env->FindClass("java/lang/Double");
        jmethodID doubleInit = env->GetMethodID(doubleClass, "<init>", "(D)V");
        jstring jkey = env->NewStringUTF(key);
        jobject jval = env->NewObject(doubleClass, doubleInit, val);
        env->CallObjectMethod(map, mapPut, jkey, jval);
        env->DeleteLocalRef(jkey);
        env->DeleteLocalRef(jval);
        env->DeleteLocalRef(doubleClass);
    };

    auto putString = [&](const char *key, const char *val) {
        jstring jkey = env->NewStringUTF(key);
        jstring jval = env->NewStringUTF(val);
        env->CallObjectMethod(map, mapPut, jkey, jval);
        env->DeleteLocalRef(jkey);
        env->DeleteLocalRef(jval);
    };

    hilum_model *model;
    hilum_context *ctx;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto mi = g_models.find(mid);
        auto ci = g_contexts.find(cid);
        if (mi == g_models.end() || ci == g_contexts.end()) {
            env->DeleteLocalRef(mapClass);
            return map;
        }
        model = mi->second;
        ctx = ci->second;
    }

    hilum_benchmark_params params = hilum_benchmark_default_params();
    params.prompt_tokens = get_int_from_map(env, options, "promptTokens", params.prompt_tokens);
    params.generate_tokens = get_int_from_map(env, options, "generateTokens", params.generate_tokens);
    params.iterations = get_int_from_map(env, options, "iterations", params.iterations);

    hilum_benchmark_result result{};
    hilum_error err = hilum_benchmark(model, ctx, params, &result);
    if (err != HILUM_OK) {
        putString("error", hilum_error_str(err));
        env->DeleteLocalRef(mapClass);
        return map;
    }

    putDouble("promptTokensPerSec", result.prompt_tokens_per_sec);
    putDouble("generatedTokensPerSec", result.generated_tokens_per_sec);
    putDouble("ttftMs", result.ttft_ms);
    putDouble("totalMs", result.total_ms);
    putDouble("iterations", (double)result.iterations);

    env->DeleteLocalRef(mapClass);
    return map;
}

// ── Vision ───────────────────────────────────────────────────────────────────

JNIEXPORT jstring JNICALL
JNI_FN(nativeLoadProjector)(JNIEnv *env, jobject thiz, jstring modelId,
                             jstring path, jobject options) {
    std::string mid = jstring_to_std(env, modelId);
    std::string pathStr = jstring_to_std(env, path);

    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_models.find(mid);
    if (it == g_models.end()) return std_to_jstring(env, "");

    hilum_mtmd_params mparams;
    mparams.use_gpu = get_bool_from_map(env, options, "use_gpu", true);
    mparams.n_threads = (uint32_t)get_int_from_map(env, options, "n_threads", 0);

    hilum_mtmd *mtmd = nullptr;
    hilum_error err = hilum_mtmd_load(it->second, pathStr.c_str(), mparams, &mtmd);
    if (err != HILUM_OK) return std_to_jstring(env, "");

    std::string mtmdId = generate_uuid();
    g_mtmd_contexts[mtmdId] = mtmd;
    return std_to_jstring(env, mtmdId);
}

JNIEXPORT jboolean JNICALL
JNI_FN(nativeSupportVision)(JNIEnv *env, jobject thiz, jstring mtmdId) {
    std::string id = jstring_to_std(env, mtmdId);
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_mtmd_contexts.find(id);
    if (it == g_mtmd_contexts.end()) return JNI_FALSE;
    return hilum_mtmd_supports_vision(it->second) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
JNI_FN(nativeFreeMtmdContext)(JNIEnv *env, jobject thiz, jstring mtmdId) {
    std::string id = jstring_to_std(env, mtmdId);
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_mtmd_contexts.find(id);
    if (it != g_mtmd_contexts.end()) {
        hilum_mtmd_free(it->second);
        g_mtmd_contexts.erase(it);
    }
}

JNIEXPORT jstring JNICALL
JNI_FN(nativeGenerateVision)(JNIEnv *env, jobject thiz, jstring modelId,
                              jstring contextId, jstring mtmdId, jstring prompt,
                              jobjectArray imageBase64s, jobject options) {
    std::string mid = jstring_to_std(env, modelId);
    std::string cid = jstring_to_std(env, contextId);
    std::string vid = jstring_to_std(env, mtmdId);
    std::string promptStr = jstring_to_std(env, prompt);

    hilum_model *model;
    hilum_context *ctx;
    hilum_mtmd *mctx;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto mi = g_models.find(mid);
        auto ci = g_contexts.find(cid);
        auto vi = g_mtmd_contexts.find(vid);
        if (mi == g_models.end() || ci == g_contexts.end() || vi == g_mtmd_contexts.end()) {
            return std_to_jstring(env, "");
        }
        model = mi->second;
        ctx = ci->second;
        mctx = vi->second;
    }

    // Decode images
    int n_images = imageBase64s ? env->GetArrayLength(imageBase64s) : 0;
    std::vector<std::vector<uint8_t>> img_data;
    std::vector<hilum_image> images;
    for (int i = 0; i < n_images; i++) {
        jstring jb64 = (jstring)env->GetObjectArrayElement(imageBase64s, i);
        auto data = decode_base64(env, jb64);
        env->DeleteLocalRef(jb64);
        if (!data.empty()) img_data.push_back(std::move(data));
    }
    for (auto &d : img_data) {
        images.push_back({d.data(), d.size()});
    }

    GenContext gc = parse_gen_context(env, options);
    std::vector<char> buf(gc.params.max_tokens * 64 + 1024);
    int32_t generated = 0;

    hilum_error err = hilum_generate_vision(model, ctx, mctx, promptStr.c_str(),
        images.data(), (int32_t)images.size(), gc.params,
        buf.data(), (int32_t)buf.size(), &generated);

    if (err != HILUM_OK) return std_to_jstring(env, "");
    return env->NewStringUTF(buf.data());
}

JNIEXPORT void JNICALL
JNI_FN(nativeStartStreamVision)(JNIEnv *env, jobject thiz, jstring modelId,
                                 jstring contextId, jstring mtmdId, jstring prompt,
                                 jobjectArray imageBase64s, jobject options) {
    std::string mid = jstring_to_std(env, modelId);
    std::string cid = jstring_to_std(env, contextId);
    std::string vid = jstring_to_std(env, mtmdId);
    std::string promptStr = jstring_to_std(env, prompt);

    hilum_model *model;
    hilum_context *ctx;
    hilum_mtmd *mctx;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto mi = g_models.find(mid);
        auto ci = g_contexts.find(cid);
        auto vi = g_mtmd_contexts.find(vid);
        if (mi == g_models.end() || ci == g_contexts.end() || vi == g_mtmd_contexts.end()) {
            jclass cls = env->GetObjectClass(thiz);
            jmethodID emitMethod = env->GetMethodID(cls, "emitToken",
                "(Ljava/lang/String;Ljava/lang/String;ZLjava/lang/String;)V");
            jstring jcid = std_to_jstring(env, cid);
            jstring jerr = env->NewStringUTF("Not found");
            env->CallVoidMethod(thiz, emitMethod, jcid, nullptr, JNI_TRUE, jerr);
            env->DeleteLocalRef(jcid);
            env->DeleteLocalRef(jerr);
            env->DeleteLocalRef(cls);
            return;
        }
        model = mi->second;
        ctx = ci->second;
        mctx = vi->second;
    }
    hilum_cancel_clear(ctx);

    int n_images = imageBase64s ? env->GetArrayLength(imageBase64s) : 0;
    std::vector<std::vector<uint8_t>> img_data;
    std::vector<hilum_image> images;
    for (int i = 0; i < n_images; i++) {
        jstring jb64 = (jstring)env->GetObjectArrayElement(imageBase64s, i);
        auto data = decode_base64(env, jb64);
        env->DeleteLocalRef(jb64);
        if (!data.empty()) img_data.push_back(std::move(data));
    }
    for (auto &d : img_data) {
        images.push_back({d.data(), d.size()});
    }

    GenContext gc = parse_gen_context(env, options);
    jobject moduleRef = env->NewGlobalRef(thiz);

    struct StreamState {
        jobject moduleRef;
        std::string ctxId;
        JavaVM *jvm;
    };
    StreamState *state = new StreamState{moduleRef, cid, g_jvm};

    hilum_generate_vision_stream(model, ctx, mctx, promptStr.c_str(),
        images.data(), (int32_t)images.size(), gc.params,
        [](const char *token, int32_t token_len, void *ud) -> bool {
            auto *s = static_cast<StreamState *>(ud);

            JNIEnv *env;
            bool detach = false;
            if (s->jvm->GetEnv((void **)&env, JNI_VERSION_1_6) == JNI_EDETACHED) {
                s->jvm->AttachCurrentThread(&env, nullptr);
                detach = true;
            }

            jclass cls = env->GetObjectClass(s->moduleRef);
            jmethodID emitMethod = env->GetMethodID(cls, "emitToken",
                "(Ljava/lang/String;Ljava/lang/String;ZLjava/lang/String;)V");
            jstring jcid = env->NewStringUTF(s->ctxId.c_str());
            jstring jtok = env->NewStringUTF(std::string(token, token_len).c_str());
            env->CallVoidMethod(s->moduleRef, emitMethod, jcid, jtok, JNI_FALSE, nullptr);
            env->DeleteLocalRef(jcid);
            env->DeleteLocalRef(jtok);
            env->DeleteLocalRef(cls);

            if (detach) s->jvm->DetachCurrentThread();
            return true;
        }, state);

    {
        jclass cls = env->GetObjectClass(thiz);
        jmethodID emitMethod = env->GetMethodID(cls, "emitToken",
            "(Ljava/lang/String;Ljava/lang/String;ZLjava/lang/String;)V");
        jstring jcid = std_to_jstring(env, cid);
        env->CallVoidMethod(thiz, emitMethod, jcid, nullptr, JNI_TRUE, nullptr);
        env->DeleteLocalRef(jcid);
        env->DeleteLocalRef(cls);
    }

    env->DeleteGlobalRef(moduleRef);
    delete state;
}

// ── Grammar ──────────────────────────────────────────────────────────────────

JNIEXPORT jstring JNICALL
JNI_FN(nativeJsonSchemaToGrammar)(JNIEnv *env, jobject thiz, jstring schemaJson) {
    std::string json = jstring_to_std(env, schemaJson);
    std::vector<char> buf(json.size() * 8 + 4096);
    int32_t len = hilum_json_schema_to_grammar(json.c_str(), buf.data(), (int32_t)buf.size());
    if (len <= 0) {
        if (len < 0) {
            buf.resize(-len);
            len = hilum_json_schema_to_grammar(json.c_str(), buf.data(), (int32_t)buf.size());
        }
        if (len <= 0) return std_to_jstring(env, "");
    }
    return env->NewStringUTF(std::string(buf.data(), len).c_str());
}

// ── Embeddings ───────────────────────────────────────────────────────────────

JNIEXPORT jint JNICALL
JNI_FN(nativeGetEmbeddingDimension)(JNIEnv *env, jobject thiz, jstring modelId) {
    std::string mid = jstring_to_std(env, modelId);
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_models.find(mid);
    if (it == g_models.end()) return 0;
    return (jint)hilum_emb_dimension(it->second);
}

JNIEXPORT jstring JNICALL
JNI_FN(nativeCreateEmbeddingContext)(JNIEnv *env, jobject thiz, jstring modelId,
                                      jobject options) {
    std::string mid = jstring_to_std(env, modelId);
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_models.find(mid);
    if (it == g_models.end()) return std_to_jstring(env, "");

    hilum_emb_params params;
    params.n_ctx        = (uint32_t)get_int_from_map(env, options, "n_ctx", 0);
    params.n_batch      = (uint32_t)get_int_from_map(env, options, "n_batch", 0);
    params.n_threads    = (uint32_t)get_int_from_map(env, options, "n_threads", 0);
    params.pooling_type = get_int_from_map(env, options, "pooling_type", -1);

    hilum_emb_ctx *ectx = nullptr;
    hilum_error err = hilum_emb_context_create(it->second, params, &ectx);
    if (err != HILUM_OK) return std_to_jstring(env, "");

    std::string ctxId = generate_uuid();
    g_emb_contexts[ctxId] = ectx;
    return std_to_jstring(env, ctxId);
}

JNIEXPORT jobject JNICALL
JNI_FN(nativeEmbed)(JNIEnv *env, jobject thiz, jstring contextId, jstring modelId,
                     jobject tokens) {
    std::string cid = jstring_to_std(env, contextId);
    std::string mid = jstring_to_std(env, modelId);
    std::vector<int32_t> tok_vec = jarray_to_int_vec(env, tokens);

    jclass arrayListClass = env->FindClass("java/util/ArrayList");
    jmethodID initMethod = env->GetMethodID(arrayListClass, "<init>", "()V");
    jmethodID addMethod = env->GetMethodID(arrayListClass, "add", "(Ljava/lang/Object;)Z");
    jobject result = env->NewObject(arrayListClass, initMethod);

    std::lock_guard<std::mutex> lock(g_mutex);
    auto ci = g_emb_contexts.find(cid);
    auto mi = g_models.find(mid);
    if (ci == g_emb_contexts.end() || mi == g_models.end()) {
        env->DeleteLocalRef(arrayListClass);
        return result;
    }

    int n_embd = hilum_emb_dimension(mi->second);
    std::vector<float> emb(n_embd);

    hilum_error err = hilum_embed(ci->second, mi->second, tok_vec.data(),
                                   (int32_t)tok_vec.size(), emb.data(), n_embd);
    if (err != HILUM_OK) {
        env->DeleteLocalRef(arrayListClass);
        return result;
    }

    jclass doubleClass = env->FindClass("java/lang/Double");
    jmethodID valueOf = env->GetStaticMethodID(doubleClass, "valueOf", "(D)Ljava/lang/Double;");
    for (int i = 0; i < n_embd; i++) {
        jobject dObj = env->CallStaticObjectMethod(doubleClass, valueOf, (jdouble)emb[i]);
        env->CallBooleanMethod(result, addMethod, dObj);
        env->DeleteLocalRef(dObj);
    }
    env->DeleteLocalRef(doubleClass);
    env->DeleteLocalRef(arrayListClass);
    return result;
}

JNIEXPORT jobject JNICALL
JNI_FN(nativeEmbedBatch)(JNIEnv *env, jobject thiz, jstring contextId, jstring modelId,
                          jobject tokenArrays) {
    std::string cid = jstring_to_std(env, contextId);
    std::string mid = jstring_to_std(env, modelId);

    jclass arrayListClass = env->FindClass("java/util/ArrayList");
    jmethodID initMethod = env->GetMethodID(arrayListClass, "<init>", "()V");
    jmethodID addMethod = env->GetMethodID(arrayListClass, "add", "(Ljava/lang/Object;)Z");
    jobject result = env->NewObject(arrayListClass, initMethod);

    jclass listClass = env->FindClass("java/util/List");
    jmethodID sizeMethod = env->GetMethodID(listClass, "size", "()I");
    jmethodID getMethod = env->GetMethodID(listClass, "get", "(I)Ljava/lang/Object;");

    std::lock_guard<std::mutex> lock(g_mutex);
    auto ci = g_emb_contexts.find(cid);
    auto mi = g_models.find(mid);
    if (ci == g_emb_contexts.end() || mi == g_models.end()) {
        env->DeleteLocalRef(arrayListClass);
        env->DeleteLocalRef(listClass);
        return result;
    }

    int n_seqs = env->CallIntMethod(tokenArrays, sizeMethod);
    int n_embd = hilum_emb_dimension(mi->second);

    std::vector<std::vector<int32_t>> tok_vecs(n_seqs);
    std::vector<const int32_t *> tok_ptrs(n_seqs);
    std::vector<int32_t> tok_counts(n_seqs);

    for (int s = 0; s < n_seqs; s++) {
        jobject arr = env->CallObjectMethod(tokenArrays, getMethod, s);
        tok_vecs[s] = jarray_to_int_vec(env, arr);
        env->DeleteLocalRef(arr);
        tok_ptrs[s] = tok_vecs[s].data();
        tok_counts[s] = (int32_t)tok_vecs[s].size();
    }

    std::vector<std::vector<float>> emb_vecs(n_seqs, std::vector<float>(n_embd));
    std::vector<float *> emb_ptrs(n_seqs);
    for (int s = 0; s < n_seqs; s++) emb_ptrs[s] = emb_vecs[s].data();

    hilum_error err = hilum_embed_batch(ci->second, mi->second,
        tok_ptrs.data(), tok_counts.data(), n_seqs, emb_ptrs.data(), n_embd);
    if (err != HILUM_OK) {
        env->DeleteLocalRef(arrayListClass);
        env->DeleteLocalRef(listClass);
        return result;
    }

    jclass doubleClass = env->FindClass("java/lang/Double");
    jmethodID dblValueOf = env->GetStaticMethodID(doubleClass, "valueOf", "(D)Ljava/lang/Double;");
    for (int s = 0; s < n_seqs; s++) {
        jobject vec = env->NewObject(arrayListClass, initMethod);
        for (int i = 0; i < n_embd; i++) {
            jobject dObj = env->CallStaticObjectMethod(doubleClass, dblValueOf, (jdouble)emb_vecs[s][i]);
            env->CallBooleanMethod(vec, addMethod, dObj);
            env->DeleteLocalRef(dObj);
        }
        env->CallBooleanMethod(result, addMethod, vec);
        env->DeleteLocalRef(vec);
    }
    env->DeleteLocalRef(doubleClass);
    env->DeleteLocalRef(arrayListClass);
    env->DeleteLocalRef(listClass);
    return result;
}

// ── Batch inference ──────────────────────────────────────────────────────────

JNIEXPORT void JNICALL
JNI_FN(nativeStartBatch)(JNIEnv *env, jobject thiz, jstring modelId, jstring contextId,
                          jobjectArray prompts, jobject options) {
    std::string mid = jstring_to_std(env, modelId);
    std::string cid = jstring_to_std(env, contextId);

    hilum_model *model;
    hilum_context *ctx;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto mi = g_models.find(mid);
        auto ci = g_contexts.find(cid);
        if (mi == g_models.end() || ci == g_contexts.end()) {
            jclass cls = env->GetObjectClass(thiz);
            jmethodID emitMethod = env->GetMethodID(cls, "emitBatchToken",
                "(Ljava/lang/String;ILjava/lang/String;ZLjava/lang/String;)V");
            jstring jcid = std_to_jstring(env, cid);
            jstring jerr = env->NewStringUTF("Not found");
            env->CallVoidMethod(thiz, emitMethod, jcid, -1, nullptr, JNI_TRUE, jerr);
            env->DeleteLocalRef(jcid);
            env->DeleteLocalRef(jerr);
            env->DeleteLocalRef(cls);
            return;
        }
        model = mi->second;
        ctx = ci->second;
    }
    hilum_cancel_clear(ctx);

    int n_seqs = prompts ? env->GetArrayLength(prompts) : 0;
    std::vector<std::string> prompt_strs(n_seqs);
    std::vector<const char *> prompt_ptrs(n_seqs);
    for (int i = 0; i < n_seqs; i++) {
        jstring jp = (jstring)env->GetObjectArrayElement(prompts, i);
        prompt_strs[i] = jstring_to_std(env, jp);
        prompt_ptrs[i] = prompt_strs[i].c_str();
        env->DeleteLocalRef(jp);
    }

    GenContext gc = parse_gen_context(env, options);
    jobject moduleRef = env->NewGlobalRef(thiz);

    struct BatchState {
        jobject moduleRef;
        std::string ctxId;
        JavaVM *jvm;
    };
    BatchState *state = new BatchState{moduleRef, cid, g_jvm};

    hilum_generate_batch(model, ctx, prompt_ptrs.data(), n_seqs, gc.params,
        [](hilum_batch_event event, void *ud) -> bool {
            auto *s = static_cast<BatchState *>(ud);

            JNIEnv *env;
            bool detach = false;
            if (s->jvm->GetEnv((void **)&env, JNI_VERSION_1_6) == JNI_EDETACHED) {
                s->jvm->AttachCurrentThread(&env, nullptr);
                detach = true;
            }

            jclass cls = env->GetObjectClass(s->moduleRef);
            jmethodID emitMethod = env->GetMethodID(cls, "emitBatchToken",
                "(Ljava/lang/String;ILjava/lang/String;ZLjava/lang/String;)V");
            jstring jcid = env->NewStringUTF(s->ctxId.c_str());

            if (event.done) {
                jstring reason = event.finish_reason
                    ? env->NewStringUTF(event.finish_reason) : env->NewStringUTF("stop");
                env->CallVoidMethod(s->moduleRef, emitMethod, jcid, event.seq_index,
                    nullptr, JNI_TRUE, reason);
                env->DeleteLocalRef(reason);
            } else {
                jstring jtok = env->NewStringUTF(std::string(event.token, event.token_len).c_str());
                env->CallVoidMethod(s->moduleRef, emitMethod, jcid, event.seq_index,
                    jtok, JNI_FALSE, nullptr);
                env->DeleteLocalRef(jtok);
            }
            env->DeleteLocalRef(jcid);
            env->DeleteLocalRef(cls);

            if (detach) s->jvm->DetachCurrentThread();
            return true;
        }, state);
    env->DeleteGlobalRef(moduleRef);
    delete state;
}

// ── Quantization ─────────────────────────────────────────────────────────────

JNIEXPORT void JNICALL
JNI_FN(nativeQuantize)(JNIEnv *env, jobject thiz, jstring inputPath,
                        jstring outputPath, jobject options) {
    std::string inPath = jstring_to_std(env, inputPath);
    std::string outPath = jstring_to_std(env, outputPath);

    hilum_quantize_params params = hilum_quantize_default_params();
    if (options) {
        params.ftype                  = get_int_from_map(env, options, "ftype", params.ftype);
        params.nthread                = get_int_from_map(env, options, "nthread", params.nthread);
        params.allow_requantize       = get_bool_from_map(env, options, "allow_requantize", params.allow_requantize);
        params.quantize_output_tensor = get_bool_from_map(env, options, "quantize_output_tensor", params.quantize_output_tensor);
        params.pure                   = get_bool_from_map(env, options, "pure", params.pure);
    }

    hilum_error err = hilum_quantize(inPath.c_str(), outPath.c_str(), params);

    jclass cls = env->GetObjectClass(thiz);
    jmethodID emitMethod = env->GetMethodID(cls, "emitQuantizeComplete",
        "(Ljava/lang/String;)V");
    jstring jerr = (err != HILUM_OK)
        ? env->NewStringUTF(hilum_error_str(err))
        : nullptr;
    env->CallVoidMethod(thiz, emitMethod, jerr);
    if (jerr) env->DeleteLocalRef(jerr);
    env->DeleteLocalRef(cls);
}

// ── Logging ──────────────────────────────────────────────────────────────────

JNIEXPORT void JNICALL
JNI_FN(nativeSetLogLevel)(JNIEnv *env, jobject thiz, jint level) {
    hilum_log_set_level(static_cast<hilum_log_level>(level));
}

JNIEXPORT void JNICALL
JNI_FN(nativeEnableLogEvents)(JNIEnv *env, jobject thiz, jboolean enabled) {
    g_log_events_enabled.store(enabled, std::memory_order_relaxed);
    if (enabled) {
        {
            std::lock_guard<std::mutex> lock(g_log_mutex);
            if (g_module_ref) {
                env->DeleteGlobalRef(g_module_ref);
            }
            g_module_ref = env->NewGlobalRef(thiz);
        }
        hilum_log_set([](hilum_log_level level, const char *text, void *) {
            if (!g_log_events_enabled.load(std::memory_order_relaxed)) return;

            JNIEnv *env;
            bool detach = false;
            if (g_jvm->GetEnv((void **)&env, JNI_VERSION_1_6) == JNI_EDETACHED) {
                g_jvm->AttachCurrentThread(&env, nullptr);
                detach = true;
            }

            std::lock_guard<std::mutex> lock(g_log_mutex);
            if (g_module_ref) {
                jclass cls = env->GetObjectClass(g_module_ref);
                jmethodID emitMethod = env->GetMethodID(cls, "emitLog",
                    "(ILjava/lang/String;)V");
                jstring jtext = env->NewStringUTF(text);
                env->CallVoidMethod(g_module_ref, emitMethod, (jint)level, jtext);
                env->DeleteLocalRef(jtext);
                env->DeleteLocalRef(cls);
            }

            if (detach) g_jvm->DetachCurrentThread();
        }, nullptr);
    } else {
        hilum_log_set(nullptr, nullptr);
        std::lock_guard<std::mutex> lock(g_log_mutex);
        if (g_module_ref) {
            env->DeleteGlobalRef(g_module_ref);
            g_module_ref = nullptr;
        }
    }
}

// ── Downloads (handled in Kotlin, JNI just provides storage path) ────────────

JNIEXPORT jstring JNICALL
JNI_FN(nativeGetModelStoragePath)(JNIEnv *env, jobject thiz, jstring filesDir) {
    std::string dir = jstring_to_std(env, filesDir);
    std::string path = dir + "/local-llm/models";
    // Create directory if needed
    std::string cmd = "mkdir -p " + path;
    system(cmd.c_str());
    return std_to_jstring(env, path);
}

} // extern "C"
