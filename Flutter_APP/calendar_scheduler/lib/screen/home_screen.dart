import 'package:calendar_scheduler/component/main_calendar.dart';
import 'package:flutter/material.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  // 선택된 날짜를 관리할 변수
  DateTime selectedDate = DateTime.utc(
    DateTime.now().year,
    DateTime.now().month,
    DateTime.now().day,
  );

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            MainCalendar(
              selectedDate: selectedDate, // 선택된 날짜 전달
              onDaySelected: onDaySelected, // 날짜가 선택시 실행할 함수
            ),
          ],
        ),
      ),
    );
  }

  // 날짜 선택시 마다 실행할 함수
  void onDaySelected(DateTime selectedDate, DateTime focusedDate) {
    // 날짜 업데이트
    setState(() {
      this.selectedDate = selectedDate;
    });
  }
}