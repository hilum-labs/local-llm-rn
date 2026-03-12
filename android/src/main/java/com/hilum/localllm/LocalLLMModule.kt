package com.hilum.localllm

import com.facebook.react.bridge.*
import com.facebook.react.modules.core.DeviceEventManagerModule
import java.io.File
import java.util.concurrent.Executors

class LocalLLMModule(reactContext: ReactApplicationContext) :
    NativeLocalLLMSpec(reactContext) {

    companion object {
        const val NAME = "LocalLLM"

        init {
            System.loadLibrary("local-llm-rn")
        }
    }

    private val executor = Executors.newSingleThreadExecutor()
    private var listenerCount = 0

    /** Allowed root for file I/O — prevents reading/writing arbitrary paths. */
    private val allowedStorageRoot: String by lazy {
        File(reactApplicationContext.filesDir, "local-llm").absolutePath
    }

    private fun isPathAllowed(path: String): Boolean {
        val resolved = File(path).canonicalPath
        return resolved.startsWith(allowedStorageRoot)
    }

    init {
        nativeInit(reactContext.applicationInfo.nativeLibraryDir)
    }

    override fun getName(): String = NAME

    // ── Event emitting (called from JNI) ────────────────────────────────────

    @Suppress("unused")
    fun emitToken(contextId: String, token: String?, done: Boolean, error: String?) {
        val params = Arguments.createMap().apply {
            putString("contextId", contextId)
            putBoolean("done", done)
            if (token != null) putString("token", token)
            if (error != null) putString("error", error)
        }
        sendEvent("onToken", params)
    }

    @Suppress("unused")
    fun emitBatchToken(contextId: String, seqIndex: Int, token: String?,
                       done: Boolean, finishReason: String?) {
        val params = Arguments.createMap().apply {
            putString("contextId", contextId)
            putInt("seqIndex", seqIndex)
            putBoolean("done", done)
            if (token != null) putString("token", token)
            if (finishReason != null) putString("finishReason", finishReason)
        }
        sendEvent("onBatchToken", params)
    }

    @Suppress("unused")
    fun emitQuantizeComplete(error: String?) {
        val params = Arguments.createMap().apply {
            if (error != null) putString("error", error) else putNull("error")
        }
        sendEvent("onQuantizeComplete", params)
    }

    @Suppress("unused")
    fun emitLog(level: Int, text: String) {
        val params = Arguments.createMap().apply {
            putInt("level", level)
            putString("text", text)
        }
        sendEvent("onLog", params)
    }

    @Suppress("unused")
    fun emitDownloadProgress(url: String, downloaded: Long, total: Long, percent: Double) {
        val params = Arguments.createMap().apply {
            putString("url", url)
            putDouble("downloaded", downloaded.toDouble())
            putDouble("total", total.toDouble())
            putDouble("percent", percent)
        }
        sendEvent("onDownloadProgress", params)
    }

    @Suppress("unused")
    fun emitDownloadComplete(url: String) {
        val params = Arguments.createMap().apply {
            putString("url", url)
        }
        sendEvent("onDownloadComplete", params)
    }

    @Suppress("unused")
    fun emitDownloadError(url: String, error: String, resumable: Boolean) {
        val params = Arguments.createMap().apply {
            putString("url", url)
            putString("error", error)
            putBoolean("resumable", resumable)
        }
        sendEvent("onDownloadError", params)
    }

    private fun sendEvent(eventName: String, params: WritableMap) {
        if (listenerCount <= 0) return
        reactApplicationContext
            .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            .emit(eventName, params)
    }

    // ── Listener management ─────────────────────────────────────────────────

    @ReactMethod
    fun addListener(eventType: String) { listenerCount++ }

    @ReactMethod
    fun removeListeners(count: Int) { listenerCount -= count }

    // ── Backend info ────────────────────────────────────────────────────────

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun backendInfo(): String = nativeBackendInfo()

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun backendVersion(): String = nativeBackendVersion()

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun apiVersion(): Double = nativeApiVersion().toDouble()

    // ── Model lifecycle ─────────────────────────────────────────────────────

    @ReactMethod
    override fun loadModel(path: String, options: ReadableMap, promise: Promise) {
        executor.execute {
            try {
                val result = nativeLoadModel(path, options.toHashMap())
                if (result.isEmpty()) {
                    promise.reject("E_MODEL_LOAD", "Failed to load model")
                } else {
                    promise.resolve(result)
                }
            } catch (e: Exception) {
                promise.reject("E_MODEL_LOAD", e.message, e)
            }
        }
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun getModelSize(modelId: String): Double = nativeGetModelSize(modelId)

    @ReactMethod
    override fun freeModel(modelId: String) { nativeFreeModel(modelId) }

    // ── Context lifecycle ───────────────────────────────────────────────────

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun createContext(modelId: String, options: ReadableMap): String =
        nativeCreateContext(modelId, options.toHashMap())

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun getContextSize(contextId: String): Double = nativeGetContextSize(contextId).toDouble()

    @ReactMethod
    override fun freeContext(contextId: String) { nativeFreeContext(contextId) }

    // ── Warmup ──────────────────────────────────────────────────────────────

    @ReactMethod
    override fun warmup(modelId: String, contextId: String, promise: Promise) {
        executor.execute {
            try {
                nativeWarmup(modelId, contextId)
                promise.resolve(null)
            } catch (e: Exception) {
                promise.reject("E_WARMUP", e.message, e)
            }
        }
    }

    // ── KV cache ────────────────────────────────────────────────────────────

    @ReactMethod
    override fun kvCacheClear(contextId: String, fromPos: Double) {
        nativeKvCacheClear(contextId, fromPos.toInt())
    }

    // ── Tokenization ────────────────────────────────────────────────────────

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun tokenize(modelId: String, text: String, addSpecial: Boolean,
                 parseSpecial: Boolean): WritableArray {
        val tokens = nativeTokenize(modelId, text, addSpecial, parseSpecial)
        val result = Arguments.createArray()
        for (t in tokens) result.pushInt(t as Int)
        return result
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun detokenize(modelId: String, tokens: ReadableArray): String {
        val tokenList = ArrayList<Int>()
        for (i in 0 until tokens.size()) tokenList.add(tokens.getInt(i))
        return nativeDetokenize(modelId, tokenList)
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun applyChatTemplate(modelId: String, messages: ReadableArray,
                          addAssistant: Boolean): String {
        // Serialize messages to JSON
        val jsonArray = org.json.JSONArray()
        for (i in 0 until messages.size()) {
            val msg = messages.getMap(i) ?: continue
            val jsonObj = org.json.JSONObject()
            val iter = msg.keySetIterator()
            while (iter.hasNextKey()) {
                val key = iter.nextKey()
                when (msg.getType(key)) {
                    ReadableType.String -> jsonObj.put(key, msg.getString(key))
                    ReadableType.Boolean -> jsonObj.put(key, msg.getBoolean(key))
                    ReadableType.Number -> jsonObj.put(key, msg.getDouble(key))
                    else -> {}
                }
            }
            jsonArray.put(jsonObj)
        }
        return nativeApplyChatTemplate(modelId, jsonArray.toString(), addAssistant)
    }

    // ── Text inference ──────────────────────────────────────────────────────

    @ReactMethod
    override fun generate(modelId: String, contextId: String, prompt: String,
                 options: ReadableMap, promise: Promise) {
        executor.execute {
            try {
                val result = nativeGenerate(modelId, contextId, prompt, options.toHashMap())
                promise.resolve(result)
            } catch (e: Exception) {
                promise.reject("E_GENERATE", e.message, e)
            }
        }
    }

    @ReactMethod
    override fun startStream(modelId: String, contextId: String, prompt: String,
                    options: ReadableMap) {
        executor.execute {
            nativeStartStream(modelId, contextId, prompt, options.toHashMap())
        }
    }

    @ReactMethod
    override fun stopStream(contextId: String) { nativeStopStream(contextId) }

    // ── Performance metrics ──────────────────────────────────────────────────

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun getPerf(contextId: String): WritableMap {
        val perf = nativeGetPerf(contextId)
        return Arguments.createMap().apply {
            putDouble("promptEvalMs", (perf["promptEvalMs"] as? Number)?.toDouble() ?: 0.0)
            putDouble("generationMs", (perf["generationMs"] as? Number)?.toDouble() ?: 0.0)
            putDouble("promptTokens", (perf["promptTokens"] as? Number)?.toDouble() ?: 0.0)
            putDouble("generatedTokens", (perf["generatedTokens"] as? Number)?.toDouble() ?: 0.0)
            putDouble("promptTokensPerSec", (perf["promptTokensPerSec"] as? Number)?.toDouble() ?: 0.0)
            putDouble("generatedTokensPerSec", (perf["generatedTokensPerSec"] as? Number)?.toDouble() ?: 0.0)
        }
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun optimalThreadCount(): Double = nativeOptimalThreadCount().toDouble()

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun benchmark(modelId: String, contextId: String, options: ReadableMap): WritableMap {
        val result = nativeBenchmark(modelId, contextId, options.toHashMap())
        return Arguments.createMap().apply {
            putDouble("promptTokensPerSec", (result["promptTokensPerSec"] as? Number)?.toDouble() ?: 0.0)
            putDouble("generatedTokensPerSec", (result["generatedTokensPerSec"] as? Number)?.toDouble() ?: 0.0)
            putDouble("ttftMs", (result["ttftMs"] as? Number)?.toDouble() ?: 0.0)
            putDouble("totalMs", (result["totalMs"] as? Number)?.toDouble() ?: 0.0)
            putDouble("iterations", (result["iterations"] as? Number)?.toDouble() ?: 0.0)
            (result["error"] as? String)?.let { putString("error", it) }
        }
    }

    // ── Vision ──────────────────────────────────────────────────────────────

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun loadProjector(modelId: String, path: String, options: ReadableMap): String =
        nativeLoadProjector(modelId, path, options.toHashMap())

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun supportVision(mtmdId: String): Boolean = nativeSupportVision(mtmdId)

    @ReactMethod
    override fun freeMtmdContext(mtmdId: String) { nativeFreeMtmdContext(mtmdId) }

    @ReactMethod
    override fun generateVision(modelId: String, contextId: String, mtmdId: String,
                       prompt: String, imageBase64s: ReadableArray,
                       options: ReadableMap, promise: Promise) {
        executor.execute {
            try {
                val images = Array(imageBase64s.size()) { imageBase64s.getString(it)!! }
                val result = nativeGenerateVision(modelId, contextId, mtmdId,
                    prompt, images, options.toHashMap())
                promise.resolve(result)
            } catch (e: Exception) {
                promise.reject("E_VISION", e.message, e)
            }
        }
    }

    @ReactMethod
    override fun startStreamVision(modelId: String, contextId: String, mtmdId: String,
                          prompt: String, imageBase64s: ReadableArray,
                          options: ReadableMap) {
        executor.execute {
            val images = Array(imageBase64s.size()) { imageBase64s.getString(it)!! }
            nativeStartStreamVision(modelId, contextId, mtmdId, prompt, images, options.toHashMap())
        }
    }

    // ── Grammar ─────────────────────────────────────────────────────────────

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun jsonSchemaToGrammar(schemaJson: String): String = nativeJsonSchemaToGrammar(schemaJson)

    // ── Embeddings ──────────────────────────────────────────────────────────

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun getEmbeddingDimension(modelId: String): Double = nativeGetEmbeddingDimension(modelId).toDouble()

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun createEmbeddingContext(modelId: String, options: ReadableMap): String =
        nativeCreateEmbeddingContext(modelId, options.toHashMap())

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun embed(contextId: String, modelId: String, tokens: ReadableArray): WritableArray {
        val tokenList = ArrayList<Int>()
        for (i in 0 until tokens.size()) tokenList.add(tokens.getInt(i))
        val embeddings = nativeEmbed(contextId, modelId, tokenList)
        val result = Arguments.createArray()
        for (v in embeddings) result.pushDouble(v as Double)
        return result
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun embedBatch(contextId: String, modelId: String, tokenArrays: ReadableArray): WritableArray {
        val outerList = ArrayList<ArrayList<Int>>()
        for (i in 0 until tokenArrays.size()) {
            val arr = tokenArrays.getArray(i) ?: continue
            val inner = ArrayList<Int>()
            for (j in 0 until arr.size()) inner.add(arr.getInt(j))
            outerList.add(inner)
        }
        val embeddings = nativeEmbedBatch(contextId, modelId, outerList)
        val result = Arguments.createArray()
        for (vec in embeddings) {
            val innerArr = Arguments.createArray()
            for (v in vec as List<*>) innerArr.pushDouble(v as Double)
            result.pushArray(innerArr)
        }
        return result
    }

    // ── Batch inference ─────────────────────────────────────────────────────

    @ReactMethod
    override fun startBatch(modelId: String, contextId: String, prompts: ReadableArray,
                   options: ReadableMap) {
        executor.execute {
            val promptArray = Array(prompts.size()) { prompts.getString(it)!! }
            nativeStartBatch(modelId, contextId, promptArray, options.toHashMap())
        }
    }

    // ── Quantization ────────────────────────────────────────────────────────

    @ReactMethod
    override fun quantize(inputPath: String, outputPath: String, options: ReadableMap) {
        executor.execute {
            nativeQuantize(inputPath, outputPath, options.toHashMap())
        }
    }

    // ── Logging ─────────────────────────────────────────────────────────────

    @ReactMethod
    override fun setLogLevel(level: Double) { nativeSetLogLevel(level.toInt()) }

    @ReactMethod
    override fun enableLogEvents(enabled: Boolean) { nativeEnableLogEvents(enabled) }

    // ── Downloads ───────────────────────────────────────────────────────────

    private val activeDownloads = java.util.concurrent.ConcurrentHashMap<String, Thread>()

    private fun executeDownload(url: String, destPath: String, resumeFrom: Long) {
        val thread = Thread {
            try {
                val destFile = File(destPath)
                destFile.parentFile?.mkdirs()

                val connection = java.net.URL(url).openConnection() as java.net.HttpURLConnection
                connection.connectTimeout = 30_000
                connection.readTimeout = 30_000

                var downloaded = resumeFrom
                if (resumeFrom > 0) {
                    connection.setRequestProperty("Range", "bytes=$resumeFrom-")
                }

                val responseCode = connection.responseCode
                // 206 = Partial Content (resume successful), 200 = full response
                val totalBytes = if (responseCode == 206) {
                    resumeFrom + connection.contentLengthLong
                } else {
                    // Server didn't honor Range — restart from scratch
                    downloaded = 0
                    connection.contentLengthLong
                }

                val inputStream = connection.inputStream
                val outputStream = if (downloaded > 0L && responseCode == 206) {
                    java.io.FileOutputStream(destFile, true) // append mode
                } else {
                    destFile.outputStream()
                }

                val buffer = ByteArray(65536)
                while (!Thread.currentThread().isInterrupted) {
                    val bytesRead = inputStream.read(buffer)
                    if (bytesRead == -1) break
                    outputStream.write(buffer, 0, bytesRead)
                    downloaded += bytesRead

                    val percent = if (totalBytes > 0) downloaded.toDouble() / totalBytes * 100.0 else 0.0
                    emitDownloadProgress(url, downloaded, totalBytes, percent)
                }

                outputStream.close()
                inputStream.close()
                connection.disconnect()
                activeDownloads.remove(url)

                if (Thread.currentThread().isInterrupted) {
                    emitDownloadError(url, "Download cancelled", true)
                } else {
                    emitDownloadComplete(url)
                }
            } catch (e: Exception) {
                activeDownloads.remove(url)
                val destFile = File(destPath)
                val resumable = destFile.exists() && destFile.length() > 0
                emitDownloadError(url, e.message ?: "Download failed", resumable)
            }
        }
        activeDownloads[url] = thread
        thread.start()
    }

    @ReactMethod
    override fun downloadModel(url: String, destPath: String) {
        executeDownload(url, destPath, 0)
    }

    @ReactMethod
    override fun resumeDownload(url: String, destPath: String) {
        val destFile = File(destPath)
        val existingBytes = if (destFile.exists()) destFile.length() else 0L
        executeDownload(url, destPath, existingBytes)
    }

    @ReactMethod
    override fun cancelDownload(url: String) {
        activeDownloads.remove(url)?.interrupt()
    }

    // ── Device capabilities ─────────────────────────────────────────────────

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun getDeviceCapabilities(): WritableMap {
        val activityManager = reactApplicationContext
            .getSystemService(android.content.Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val memInfo = android.app.ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)

        val result = Arguments.createMap().apply {
            putDouble("totalRAM", memInfo.totalMem.toDouble())
            putDouble("availableRAM", memInfo.availMem.toDouble())
            putString("gpuName", "Vulkan GPU") // Updated at runtime if Vulkan init succeeds
            putString("chipset", android.os.Build.SOC_MODEL)
            putString("androidVersion", android.os.Build.VERSION.RELEASE)
            putBoolean("isLowPowerMode",
                (reactApplicationContext.getSystemService(android.content.Context.POWER_SERVICE)
                    as android.os.PowerManager).isPowerSaveMode)
        }
        return result
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun getModelStoragePath(): String {
        val dir = File(reactApplicationContext.filesDir, "local-llm/models")
        if (!dir.exists()) dir.mkdirs()
        return dir.absolutePath
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun fileExists(path: String): Boolean = File(path).exists()

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun getFileSize(path: String): Double = File(path).length().toDouble()

    @ReactMethod(isBlockingSynchronousMethod = true)
    override fun readTextFile(path: String): String? {
        if (!isPathAllowed(path)) return null
        val file = File(path)
        if (!file.exists()) return null
        return file.readText()
    }

    @ReactMethod
    override fun writeTextFile(path: String, content: String) {
        if (!isPathAllowed(path)) return
        val file = File(path)
        file.parentFile?.mkdirs()
        file.writeText(content)
    }

    @ReactMethod
    override fun removePath(path: String) {
        if (!isPathAllowed(path)) return
        val file = File(path)
        if (file.exists()) {
            file.deleteRecursively()
        }
    }

    @ReactMethod
    override fun sha256File(path: String, promise: Promise) {
        executor.execute {
            try {
                val file = File(path)
                if (!file.exists()) {
                    promise.reject("E_FILE_NOT_FOUND", "File not found: $path")
                    return@execute
                }
                val digest = java.security.MessageDigest.getInstance("SHA-256")
                file.inputStream().buffered().use { stream ->
                    val buffer = ByteArray(65536)
                    while (true) {
                        val bytesRead = stream.read(buffer)
                        if (bytesRead == -1) break
                        digest.update(buffer, 0, bytesRead)
                    }
                }
                val hex = digest.digest().joinToString("") { "%02x".format(it) }
                promise.resolve(hex)
            } catch (e: Exception) {
                promise.reject("E_SHA256", e.message, e)
            }
        }
    }

    // ── Native method declarations ──────────────────────────────────────────

    private external fun nativeInit(nativeLibDir: String)
    private external fun nativeBackendInfo(): String
    private external fun nativeBackendVersion(): String
    private external fun nativeApiVersion(): Int
    private external fun nativeLoadModel(path: String, options: HashMap<String, Any?>): String
    private external fun nativeGetModelSize(modelId: String): Double
    private external fun nativeFreeModel(modelId: String)
    private external fun nativeCreateContext(modelId: String, options: HashMap<String, Any?>): String
    private external fun nativeGetContextSize(contextId: String): Int
    private external fun nativeFreeContext(contextId: String)
    private external fun nativeWarmup(modelId: String, contextId: String)
    private external fun nativeKvCacheClear(contextId: String, fromPos: Int)
    private external fun nativeTokenize(modelId: String, text: String,
                                         addSpecial: Boolean, parseSpecial: Boolean): List<Any>
    private external fun nativeDetokenize(modelId: String, tokens: ArrayList<Int>): String
    private external fun nativeApplyChatTemplate(modelId: String, messagesJson: String,
                                                  addAssistant: Boolean): String
    private external fun nativeGenerate(modelId: String, contextId: String,
                                         prompt: String, options: HashMap<String, Any?>): String
    private external fun nativeStartStream(modelId: String, contextId: String,
                                            prompt: String, options: HashMap<String, Any?>)
    private external fun nativeStopStream(contextId: String)
    private external fun nativeGetPerf(contextId: String): HashMap<String, Any?>
    private external fun nativeOptimalThreadCount(): Int
    private external fun nativeBenchmark(modelId: String, contextId: String,
                                          options: HashMap<String, Any?>): HashMap<String, Any?>
    private external fun nativeLoadProjector(modelId: String, path: String,
                                              options: HashMap<String, Any?>): String
    private external fun nativeSupportVision(mtmdId: String): Boolean
    private external fun nativeFreeMtmdContext(mtmdId: String)
    private external fun nativeGenerateVision(modelId: String, contextId: String,
                                               mtmdId: String, prompt: String,
                                               imageBase64s: Array<String>,
                                               options: HashMap<String, Any?>): String
    private external fun nativeStartStreamVision(modelId: String, contextId: String,
                                                  mtmdId: String, prompt: String,
                                                  imageBase64s: Array<String>,
                                                  options: HashMap<String, Any?>)
    private external fun nativeJsonSchemaToGrammar(schemaJson: String): String
    private external fun nativeGetEmbeddingDimension(modelId: String): Int
    private external fun nativeCreateEmbeddingContext(modelId: String,
                                                      options: HashMap<String, Any?>): String
    private external fun nativeEmbed(contextId: String, modelId: String,
                                      tokens: ArrayList<Int>): List<Any>
    private external fun nativeEmbedBatch(contextId: String, modelId: String,
                                           tokenArrays: ArrayList<ArrayList<Int>>): List<Any>
    private external fun nativeStartBatch(modelId: String, contextId: String,
                                           prompts: Array<String>,
                                           options: HashMap<String, Any?>)
    private external fun nativeQuantize(inputPath: String, outputPath: String,
                                         options: HashMap<String, Any?>)
    private external fun nativeSetLogLevel(level: Int)
    private external fun nativeEnableLogEvents(enabled: Boolean)
    private external fun nativeGetModelStoragePath(filesDir: String): String
}
