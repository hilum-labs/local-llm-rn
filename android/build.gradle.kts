import org.jetbrains.kotlin.gradle.dsl.KotlinAndroidProjectExtension
import java.util.Properties

buildscript {
    repositories { mavenCentral(); google() }
}

fun resolveAndroidSdkDir(project: Project): File? {
    val localProperties = project.rootProject.file("local.properties")
    if (localProperties.exists()) {
        val properties = Properties()
        localProperties.inputStream().use(properties::load)
        properties.getProperty("sdk.dir")?.let { return File(it) }
    }

    return sequenceOf("ANDROID_SDK_ROOT", "ANDROID_HOME")
        .mapNotNull { System.getenv(it) }
        .map(::File)
        .firstOrNull(File::exists)
}

fun ensureCmake(project: Project, version: String) {
    val sdkDir = resolveAndroidSdkDir(project) ?: return
    val cmakeDir = sdkDir.resolve("cmake/$version")
    if (cmakeDir.exists()) {
        println("local-llm-rn: CMake $version found at ${cmakeDir.absolutePath}")
        return
    }

    // Locate sdkmanager
    val sdkmanager = sequenceOf(
        sdkDir.resolve("cmdline-tools/latest/bin/sdkmanager"),
        sdkDir.resolve("cmdline-tools/bin/sdkmanager"),
        sdkDir.resolve("tools/bin/sdkmanager"),
    ).firstOrNull { it.exists() } ?: return

    println("local-llm-rn: Installing CMake $version via sdkmanager…")
    val process = ProcessBuilder(sdkmanager.absolutePath, "cmake;$version")
        .redirectErrorStream(true)
        .start()
    process.inputStream.bufferedReader().forEachLine { println(it) }
    val exitCode = process.waitFor()
    if (exitCode != 0) {
        println("WARNING: sdkmanager exited with code $exitCode — CMake $version may not be installed")
    }
}

fun resolveGlslc(project: Project): String? {
    val executableName = if (System.getProperty("os.name").startsWith("Windows")) "glslc.exe" else "glslc"
    val sdkDir = resolveAndroidSdkDir(project)

    val ndkRoots = buildList {
        sequenceOf("ANDROID_NDK_ROOT", "ANDROID_NDK_HOME")
            .mapNotNull { System.getenv(it) }
            .map(::File)
            .filter(File::exists)
            .forEach(::add)

        sdkDir?.resolve("ndk")?.listFiles()
            ?.sortedByDescending { it.name }
            ?.forEach(::add)

        sdkDir?.resolve("ndk-bundle")
            ?.takeIf(File::exists)
            ?.let(::add)
    }

    return ndkRoots.asSequence()
        .map { it.resolve("shader-tools") }
        .filter(File::exists)
        .flatMap { shaderTools ->
            shaderTools.listFiles()
                ?.asSequence()
                ?.map { it.resolve(executableName) }
                ?: emptySequence()
        }
        .firstOrNull(File::exists)
        ?.absolutePath
}

plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("com.facebook.react")
}

react {
    root = file("..")
    reactNativeDir = file("../../react-native")
    codegenDir = file("../../@react-native/codegen")
    cliFile = file("../../react-native/cli.js")
    jsRootDir = file("../src")
    libraryName = "LocalLLMSpec"
    codegenJavaPackageName = "com.hilum.localllm"
}

// Auto-install CMake 4.1.2 if missing (plug-and-play for consumers).
ensureCmake(project, "4.1.2")

android {
    namespace = "com.hilum.localllm"
    compileSdk = 35

    defaultConfig {
        // API 29 = Android 10+. Vulkan 1.1 (required by the engine) is available
        // from API 29. Devices without Vulkan fall back to CPU inference.
        minSdk = 29
        ndk { abiFilters += listOf("arm64-v8a") }

        externalNativeBuild {
            cmake {
                // Resolve glslc from the Android SDK/NDK installation without
                // relying on AGP's ndkDirectory during library configuration.
                val glslc = resolveGlslc(project)

                arguments += listOfNotNull(
                    "-Wno-dev",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DBUILD_SHARED_LIBS=ON",
                    "-DLLAMA_BUILD_COMMON=ON",
                    "-DLLAMA_OPENSSL=OFF",
                    // CPU variant dispatch (2-4x speedup on modern ARM)
                    "-DGGML_NATIVE=OFF",
                    "-DGGML_BACKEND_DL=ON",
                    "-DGGML_CPU_ALL_VARIANTS=ON",
                    "-DGGML_LLAMAFILE=OFF",
                    // Vulkan GPU + Adreno optimizations
                    glslc?.let { "-DVulkan_GLSLC_EXECUTABLE=$it" },
                    "-DGGML_VULKAN=ON",
                    "-DGGML_VULKAN_VMA=ON",
                    "-DGGML_VULKAN_BUILD_ADRENO_SHADERS=ON",
                    // Disable unneeded targets
                    "-DLLAMA_BUILD_TOOLS=OFF",
                    "-DLLAMA_BUILD_TESTS=OFF",
                    "-DLLAMA_BUILD_EXAMPLES=OFF",
                    "-DLLAMA_BUILD_SERVER=OFF",
                )
            }
        }
    }

    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "4.1.2"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

}

configure<KotlinAndroidProjectExtension> {
    jvmToolchain(17)
}

dependencies {
    implementation("com.facebook.react:react-android")
}
