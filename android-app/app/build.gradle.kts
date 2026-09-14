plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

layout.buildDirectory.set(file("build2"))

android {
    namespace = "com.voice.shield"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.voice.shield"
        minSdk = 26           // Android 8 Oreo — minimum for foreground services
        targetSdk = 34        // Android 14
        versionCode = 1
        versionName = "1.0.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.activity:activity-ktx:1.8.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    // AndroidX Core & AppCompat
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")

    // Kotlin Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // OkHttp (WebSocket client)
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // Material (for notification icons etc.)
    implementation("com.google.android.material:material:1.11.0")
}
