import 'package:calendar_scheduler/component/main_calendar.dart';
import 'package:calendar_scheduler/component/schedule_bottom_sheet.dart';
import 'package:calendar_scheduler/component/schedule_card.dart';
import 'package:calendar_scheduler/component/today_banner.dart';
import 'package:calendar_scheduler/const/colors.dart';
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
      // HomeScreen에서 FloatingActionButtom을 누르면 ScheduleBottomSheet 화면 추가
      floatingActionButton: FloatingActionButton( // 새 일정 버튼 추가
        backgroundColor: PRIMARY_COLOR,
        onPressed: () {
          showModalBottomSheet( // ScheduleBottomSheet 열기
            context: context, // BuildContext를 전달, 메인 위젯이 지금 트리의 어디에 있는지 알려주는 좌표 같은 역할
            isDismissible: true, // 배경 탭 했을시 ScheduleBottomSheet 닫기
            builder: (_) => ScheduleBottomSheet(), // ScheduleBottomSheet() 불러오기
            // ScheduleBottomSheet의 높이를 화면의 최대 높이로 정의 스크롤 가능하게 변경
            isScrollControlled: true,
          );
        },
        child: Icon(
          Icons.add,
        ),
      ),
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