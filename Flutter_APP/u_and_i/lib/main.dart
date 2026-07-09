import 'package:u_and_i/screen/home_screen.dart';
import 'package:flutter/material.dart';

void main() {
  runApp(
    MaterialApp(
      theme: ThemeData(
        fontFamily: 'sunflower', // 폰트 패밀리 설정
        textTheme: TextTheme(
          headlineLarge: TextStyle(
            color: Colors.white,
            fontSize: 32.0, // 폰트 크기 설정
            fontWeight: FontWeight.w400, // 폰트 굵기 설정
            fontFamily: 'parisienne',
             // 폰트 색상 설정
          ),
          titleLarge: TextStyle(
            color: Colors.white,
            fontSize: 22.0, // 폰트 크기 설정
            fontWeight: FontWeight.w400,
          ),
          bodyMedium: TextStyle(
            color: Colors.white,
            fontSize: 14.0, // 폰트 크기 설정
            fontWeight: FontWeight.w400,
          ),
          labelSmall: TextStyle(
            color: Colors.white,
            fontSize: 12.0, // 폰트 크기 설정
            fontWeight: FontWeight.w400,
          ),
        ),
      ),
      home: HomeScreen(),
    ),
  );
}