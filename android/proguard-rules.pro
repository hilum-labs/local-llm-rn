# Keep all JNI-facing classes and methods
-keep class com.hilum.localllm.** { *; }

# Keep native methods from being stripped
-keepclassmembers class com.hilum.localllm.LocalLLMModule {
    native <methods>;
    void emitToken(...);
    void emitBatchToken(...);
    void emitDownloadProgress(...);
    void emitDownloadComplete(...);
    void emitDownloadError(...);
    void emitQuantizeComplete(...);
    void emitLog(...);
}
