package com.example.driverguardian.ui.theme

import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.os.Build
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val DarkDriverColorScheme: ColorScheme = darkColorScheme(
    primary = SafeGreen,
    secondary = AccentBlue,
    tertiary = WarningYellow,
    background = DarkAppBackground,
    surface = DarkAppSurface,
    surfaceVariant = DarkAppSurfaceHigh,
    error = DangerRed,
    outline = DarkAppBorder,
    outlineVariant = DarkAppBorder,
    onPrimary = DarkAppBackground,
    onSecondary = DarkTextPrimary,
    onTertiary = DarkAppBackground,
    onBackground = DarkTextPrimary,
    onSurface = DarkTextPrimary,
    onSurfaceVariant = DarkTextSecondary,
    onError = DarkTextPrimary
)

private val LightDriverColorScheme: ColorScheme = lightColorScheme(
    primary = SafeGreen,
    secondary = AccentBlue,
    tertiary = WarningYellow,
    background = LightAppBackground,
    surface = LightAppSurface,
    surfaceVariant = LightAppSurfaceHigh,
    error = DangerRed,
    outline = LightAppBorder,
    outlineVariant = LightAppBorder,
    onPrimary = LightAppBackground,
    onSecondary = LightTextPrimary,
    onTertiary = LightAppBackground,
    onBackground = LightTextPrimary,
    onSurface = LightTextPrimary,
    onSurfaceVariant = LightTextSecondary,
    onError = LightTextPrimary
)

@Composable
fun DriverGuardianTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) DarkDriverColorScheme else LightDriverColorScheme
    val view = LocalView.current

    if (!view.isInEditMode) {
        SideEffect {
            val window = view.context.findActivity()?.window ?: return@SideEffect
            val background = colorScheme.background.toArgb()
            window.statusBarColor = background
            window.navigationBarColor = background
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                window.isNavigationBarContrastEnforced = false
            }
            WindowCompat.getInsetsController(window, view).apply {
                isAppearanceLightStatusBars = !darkTheme
                isAppearanceLightNavigationBars = !darkTheme
            }
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = AppTypography
    ) {
        CompositionLocalProvider(
            LocalContentColor provides colorScheme.onBackground,
            content = content
        )
    }
}

private tailrec fun Context.findActivity(): Activity? = when (this) {
    is Activity -> this
    is ContextWrapper -> baseContext.findActivity()
    else -> null
}
