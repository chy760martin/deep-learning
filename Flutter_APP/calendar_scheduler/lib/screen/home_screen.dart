import 'package:calendar_scheduler/component/main_calendar.dart';
import 'package:calendar_scheduler/component/schedule_card.dart';
import 'package:calendar_scheduler/component/today_banner.dart';
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
            TodayBanner( // 배너 추가
              selectedDate: selectedDate, 
              count: 0,
            ),
            ScheduleCard( // 일정 카드 추가
              startTime: 12, 
              endTime: 14, 
              content: '프로그래밍 공부',
            ),
          ],
        ),
      ),
    );
  }

  // 날짜 선택시 마다 실행할 함수,TableCalendar 파라미터 선택날짜(selectedDate), 기준날짜(focusedDate)
  void onDaySelected(DateTime selectedDate, DateTime focusedDate) {
    // 날짜 업데이트
    setState(() {
      this.selectedDate = selectedDate;
    });
  }
}