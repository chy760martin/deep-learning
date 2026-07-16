import 'package:flutter/material.dart';
import 'package:calendar_scheduler/screen/home_screen.dart';
import 'package:intl/date_symbol_data_local.dart';


void main() async {
  // 플러터 프레임워크가 준비될 때까지 대기, 즉 플러터 엔진과 위젯 바인딩 초기화, await 비동기 작업 하기 전에 호출
  WidgetsFlutterBinding.ensureInitialized();

  // intl 패키지 초기화(다국어화)
  await initializeDateFormatting();
  
  runApp(
    MaterialApp(
      home: HomeScreen(),
    )
  );
}