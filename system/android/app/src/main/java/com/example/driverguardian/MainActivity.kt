package com.example.driverguardian

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.example.driverguardian.ui.navigation.DriverGuardianApp
import com.example.driverguardian.ui.theme.DriverGuardianTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            DriverGuardianTheme {
                DriverGuardianApp()
            }
        }
    }
}
