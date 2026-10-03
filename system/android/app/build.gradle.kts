import org.jetbrains.kotlin.gradle.dsl.JvmTarget

val driverGuardianApiBaseUrl = providers.gradleProperty("DRIVER_GUARDIAN_API_BASE_URL")
    .orElse("http://10.0.2.2:8000/")
    .get()
    .let { if (it.endsWith("/")) it else "$it/" }
val driverGuardianApiBaseUrlLiteral = "\"" + driverGuardianApiBaseUrl
    .replace("\\", "\\\\")
    .replace("\"", "\\\"") + "\""

val googleServerClientId = providers.gradleProperty("GOOGLE_SERVER_CLIENT_ID")
    .orElse("")
    .get()
val googleServerClientIdLiteral = "\"" + googleServerClientId
    .replace("\\", "\\\\")
    .replace("\"", "\\\"") + "\""

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.example.driverguardian"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.driverguardian"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "1.0.0"
        buildConfigField("String", "API_BASE_URL", driverGuardianApiBaseUrlLiteral)
        buildConfigField("String", "GOOGLE_SERVER_CLIENT_ID", googleServerClientIdLiteral)
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    val cameraXVersion = "1.5.3"

    implementation(platform("androidx.compose:compose-bom:2025.11.00"))
    implementation("androidx.activity:activity-compose:1.12.0")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.10.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.10.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.10.0")
    implementation("androidx.navigation:navigation-compose:2.9.6")
    implementation("androidx.camera:camera-camera2:$cameraXVersion")
    implementation("androidx.camera:camera-core:$cameraXVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraXVersion")
    implementation("androidx.camera:camera-view:$cameraXVersion")
    implementation("com.google.mediapipe:tasks-vision:1.0.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.30.0")
    implementation("com.squareup.retrofit2:retrofit:2.11.0")
    implementation("com.squareup.retrofit2:converter-gson:2.11.0")

    // Google Authentication & Credential Manager
    implementation("androidx.credentials:credentials:1.3.0")
    implementation("androidx.credentials:credentials-play-services-auth:1.3.0")
    implementation("com.google.android.libraries.identity.googleid:googleid:1.1.1")

    // Secure Storage (Android Keystore backed)
    implementation("androidx.security:security-crypto:1.1.0-alpha06")

    debugImplementation("androidx.compose.ui:ui-tooling")
    testImplementation("junit:junit:4.13.2")
    testImplementation("com.squareup.okhttp3:mockwebserver:4.12.0")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.2")
}
