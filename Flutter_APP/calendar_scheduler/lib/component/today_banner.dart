import 'package:calendar_scheduler/const/colors.dart';
import 'package:flutter/material.dart';

// 오늘 날짜 보여주는 TodayBanner 위젯
class TodayBanner extends StatelessWidget {
  final DateTime selectedDate; // 선택된 날짜 저장
  final int count; // 일정 개수 저장

  const TodayBanner({
    required this.selectedDate, // 외부로 부터 선택날짜 받아옴
    required this.count, // 외부로 부터 일정 개수 받아옴
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    final textStyle = TextStyle(
      fontWeight: FontWeight.w600,
      color: Colors.white,
    );

    return Container(
      color: PRIMARY_COLOR,
      child: Padding(
        padding: EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text( // 년 월 일 형태로 표시
              '${selectedDate.year}년 ${selectedDate.month}월 ${selectedDate.day}일',
              style: textStyle,
            ),
            Text(
              '${count}개', // 일정 개수 표시
              style: textStyle,
            ),
          ],
        ),
      ),
    );
  }
}