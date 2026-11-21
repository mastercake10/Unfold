import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// OpenCV Android SDK and Eigen locations come from local.properties (see README)
val localProps = Properties().apply {
    rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use { load(it) }
}
val opencvDir = localProps.getProperty("opencv.dir") ?: "/opt/opencv-android/sdk/native/jni"
val eigenDir = localProps.getProperty("eigen.dir") ?: "/opt/eigen-3.4.0"
// release key: environment (CI) or local.properties; without it release builds use the debug key
fun secret(name: String): String? = System.getenv(name) ?: localProps.getProperty(name)
val releaseKeystore = secret("RELEASE_KEYSTORE")

android {
    namespace = "de.spaceio.docscan"
    compileSdk = 35
    buildToolsVersion = "35.0.0"
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "de.spaceio.docscan"
        minSdk = 29
        targetSdk = 35
        versionCode = 2
        versionName = "0.2"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake {
                arguments += listOf("-DOpenCV_DIR=$opencvDir", "-DEIGEN_DIR=$eigenDir", "-DANDROID_STL=c++_shared")
                cppFlags += "-O3"
            }
        }
    }

    signingConfigs {
        create("release") {
            if (releaseKeystore != null) {
                storeFile = file(releaseKeystore)
                storePassword = secret("RELEASE_KEYSTORE_PASSWORD")
                keyAlias = "unfold"
                keyPassword = storePassword  // PKCS12: one password for store and key
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName(if (releaseKeystore != null) "release" else "debug")
        }
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }
}

dependencies {
    val camerax = "1.4.1"
    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-core")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")
}
