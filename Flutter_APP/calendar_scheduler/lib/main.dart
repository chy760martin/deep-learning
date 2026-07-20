import 'package:calendar_scheduler/database/drift_database.dart';
import 'package:flutter/material.dart';
import 'package:calendar_scheduler/screen/home_screen.dart';
import 'package:get_it/get_it.dart';
import 'package:intl/date_symbol_data_local.dart';


void main() async {
  // 플러터 프레임워크가 준비될 때까지 대기, 즉 플러터 엔진과 위젯 바인딩 초기화, await 비동기 작업 하기 전에 호출
  WidgetsFlutterBinding.ensureInitialized();

  // intl 패키지 초기화(다국어화)
  await initializeDateFormatting();

  // 데이터베이스 생성
  final database = LocalDatabase();

  // GetIt에 데이터베이스 변수 주입하기, 전역에서 사용
  GetIt.I.registerSingleton<LocalDatabase>(database);
  
  runApp(
    MaterialApp(
      home: HomeScreen(),
    )
  );
}