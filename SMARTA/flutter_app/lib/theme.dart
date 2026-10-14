import 'package:flutter/material.dart';

/// Shared warm editorial palette used by every SMARTA destination.
class AppTheme {
  static const Color background = Color(0xffF8EBDD);
  static const Color sidebar = Color(0xffF6E3CD);
  static const Color card = Color(0xffFFF8F1);
  static const Color card2 = Color(0xffF7EBDD);
  static const Color accent = Color(0xff2E806F);
  static const Color accentSoft = Color(0xffDDF0E8);
  static const Color botanical = Color(0xff478454);
  static const Color warning = Color(0xff9B651C);
  static const Color warningSurface = Color(0xffF8E7C8);
  static const Color danger = Color(0xffA84635);
  static const Color dangerSurface = Color(0xffF7E0D7);
  static const Color success = Color(0xff347650);
  static const Color successSurface = Color(0xffE0ECD8);
  static const Color textPrimary = Color(0xff3A241C);
  static const Color textSecondary = Color(0xff78685D);
  static const Color border = Color(0xffE9D9C8);

  static ThemeData get darkTheme => lightTheme;

  static ThemeData get lightTheme {
    final scheme = ColorScheme.fromSeed(
      seedColor: accent,
      brightness: Brightness.light,
      primary: accent,
      onPrimary: Colors.white,
      secondary: botanical,
      surface: card,
      onSurface: textPrimary,
      error: danger,
      onError: Colors.white,
    );
    final base = ThemeData(
      useMaterial3: true,
      colorScheme: scheme,
      scaffoldBackgroundColor: background,
      fontFamily: 'Arial',
      dividerColor: border,
      canvasColor: card,
      textTheme: const TextTheme(
        displayLarge: TextStyle(
            color: textPrimary,
            fontSize: 52,
            fontWeight: FontWeight.w800,
            height: 1.04,
            letterSpacing: -1.8),
        displayMedium: TextStyle(
            color: textPrimary,
            fontSize: 42,
            fontWeight: FontWeight.w800,
            height: 1.08,
            letterSpacing: -1.2),
        headlineLarge: TextStyle(
            color: textPrimary,
            fontSize: 34,
            fontWeight: FontWeight.w800,
            height: 1.1,
            letterSpacing: -.8),
        headlineMedium: TextStyle(
            color: textPrimary,
            fontSize: 28,
            fontWeight: FontWeight.w700,
            height: 1.15,
            letterSpacing: -.45),
        headlineSmall: TextStyle(
            color: textPrimary,
            fontSize: 23,
            fontWeight: FontWeight.w700,
            height: 1.2),
        titleLarge: TextStyle(
            color: textPrimary, fontSize: 19, fontWeight: FontWeight.w700),
        titleMedium: TextStyle(
            color: textPrimary, fontSize: 16, fontWeight: FontWeight.w600),
        titleSmall: TextStyle(
            color: textPrimary, fontSize: 14, fontWeight: FontWeight.w600),
        bodyLarge: TextStyle(color: textPrimary, fontSize: 15, height: 1.55),
        bodyMedium: TextStyle(color: textSecondary, fontSize: 14, height: 1.55),
        bodySmall: TextStyle(color: textSecondary, fontSize: 12, height: 1.45),
        labelLarge: TextStyle(
            color: textPrimary, fontSize: 14, fontWeight: FontWeight.w700),
        labelMedium: TextStyle(
            color: textSecondary, fontSize: 12, fontWeight: FontWeight.w600),
        labelSmall: TextStyle(
            color: textSecondary, fontSize: 11, fontWeight: FontWeight.w600),
      ),
    );

    return base.copyWith(
      appBarTheme: const AppBarTheme(
        backgroundColor: sidebar,
        foregroundColor: textPrimary,
        elevation: 0,
        surfaceTintColor: Colors.transparent,
      ),
      cardTheme: CardThemeData(
        color: card,
        elevation: 0,
        surfaceTintColor: Colors.transparent,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(22),
          side: const BorderSide(color: border),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: Colors.white,
        hintStyle: const TextStyle(color: textSecondary),
        labelStyle: const TextStyle(color: textSecondary, fontSize: 13),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 15),
        border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(14),
            borderSide: const BorderSide(color: border)),
        enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(14),
            borderSide: const BorderSide(color: border)),
        focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(14),
            borderSide: const BorderSide(color: accent, width: 1.5)),
        errorBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(14),
            borderSide: const BorderSide(color: danger)),
        focusedErrorBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(14),
            borderSide: const BorderSide(color: danger, width: 1.5)),
      ),
      iconTheme: const IconThemeData(color: textSecondary),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: accent,
          foregroundColor: Colors.white,
          disabledBackgroundColor: border,
          disabledForegroundColor: textSecondary,
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 15),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
          textStyle: const TextStyle(fontWeight: FontWeight.w700),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: accent,
          foregroundColor: Colors.white,
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 15),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
          textStyle: const TextStyle(fontWeight: FontWeight.w700),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: textPrimary,
          side: const BorderSide(color: border),
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
          textStyle: const TextStyle(fontWeight: FontWeight.w600),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
            foregroundColor: accent,
            textStyle: const TextStyle(fontWeight: FontWeight.w700)),
      ),
      chipTheme: base.chipTheme.copyWith(
        backgroundColor: card2,
        side: const BorderSide(color: border),
        labelStyle:
            const TextStyle(color: textPrimary, fontWeight: FontWeight.w500),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      ),
      navigationBarTheme: const NavigationBarThemeData(
        backgroundColor: sidebar,
        indicatorColor: accentSoft,
        labelTextStyle: WidgetStatePropertyAll(
            TextStyle(color: textPrimary, fontWeight: FontWeight.w600)),
      ),
      scrollbarTheme: ScrollbarThemeData(
        thumbColor: WidgetStateProperty.all(accent.withValues(alpha: .42)),
        radius: const Radius.circular(20),
      ),
    );
  }
}
