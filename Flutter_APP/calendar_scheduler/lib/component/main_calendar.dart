// 달력 구현
import 'package:calendar_scheduler/const/colors.dart';
import 'package:flutter/material.dart';
import 'package:table_calendar/table_calendar.dart';

class MainCalendar extends StatelessWidget {
  // selectedDate, onDaySelected 전달 받은 파라미터를 selectedDate, onDaySelected 변수에 저장
  final OnDaySelected onDaySelected; // 날짜 선택시 실행할 함수 변수에 저장
  final DateTime selectedDate; // 선택된 날짜 변수에 저장

  // HomeScreen 클래스 -> MainCalendar 생성자로 selectedDate, onDaySelected 전달
  MainCalendar({
    required this.onDaySelected, // 콜백 함수를 전달 받는다
    required this.selectedDate,  // 선택 날짜를 전달 받는다
  });

  @override
  Widget build(BuildContext context) {
    return TableCalendar(
      onDaySelected: onDaySelected, // 날짜 선택시 실행할 함수
      
      // 선택된 날짜를 구분할 로직, 년/월/일 true 값 반환
      selectedDayPredicate: (date) =>
        date.year == selectedDate.year &&
        date.month == selectedDate.month &&
        date.day == selectedDate.day,
      firstDay: DateTime(1800, 1, 1), // 첫째 날
      lastDay: DateTime(3000, 1, 1), // 마지막 날
      focusedDay: DateTime.now(), // 화면에 보여지는 날
  
      headerStyle: HeaderStyle( // 달력 최상단 스타일
        titleCentered: true, // 제목 중앙에 위치
        formatButtonVisible: false, // 달력 크기 선택 옵션 없애기
        titleTextStyle: TextStyle( // 제목 글꼴
          fontWeight: FontWeight.w700,
          fontSize: 16.0,
        ),
      ),
      calendarStyle: CalendarStyle(
        isTodayHighlighted: false,
        defaultDecoration: BoxDecoration( // 기본 날짜 스타일
          borderRadius: BorderRadius.circular(6.0),
          color: LIGHT_GREY_COLOR,
        ),
        weekendDecoration: BoxDecoration( // 주말 날짜 스타일
          borderRadius: BorderRadius.circular(6.0),
          color: LIGHT_GREY_COLOR,
        ),
        selectedDecoration: BoxDecoration( // 선택된 날짜 스타일
          borderRadius: BorderRadius.circular(6.0),
          border: Border.all(
            color: PRIMARY_COLOR,
            width: 1.0,
          ),
        ),
        defaultTextStyle: TextStyle( // 기본 글꼴
          fontWeight: FontWeight.w600,
          color: DARK_GREY_COLOR,
        ),
        weekendTextStyle: TextStyle( // 주말 글꼴
          fontWeight: FontWeight.w600,
          color: DARK_GREY_COLOR,
        ),
        selectedTextStyle: TextStyle( // 선택된 날짜 글꼴
          fontWeight: FontWeight.w600,
          color: PRIMARY_COLOR,
        ),
      ),
    );
  }
}