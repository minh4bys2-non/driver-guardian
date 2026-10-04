package com.example.driverguardian.ui.monitoring

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.driverguardian.ai.monitoring.DriverMonitoringEngine

class MonitoringViewModelFactory(private val context: Context) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        require(modelClass.isAssignableFrom(MonitoringViewModel::class.java))
        return MonitoringViewModel(DriverMonitoringEngine(context.applicationContext)) as T
    }
}
