import 'package:flutter/material.dart';

import 'theme.dart';
import 'screens/home_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  runApp(const SmartaApp());
}

class SmartaApp extends StatelessWidget {
  const SmartaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SMARTA',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme,
      home: const HomeScreen(),
    );
  }
}
