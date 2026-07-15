import 'package:calendar_scheduler/const/colors.dart';
import 'package:flutter/material.dart';

// ScheduleCard 위젯
class ScheduleCard extends StatelessWidget {
  final int startTime; // 시작 시간 저장
  final int endTime; // 종료 시간 저장
  final String content; // 일정 내용 저장

  const ScheduleCard({
    required this.startTime, // 외부로 부터 시작 시간을 받아옴
    required this.endTime, // 외부로 부터 종료 시간을 받아옴
    required this.content, // 외부로 부터 일정 내용을 받아옴
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        border: Border.all(
          width: 1.0,
          color: PRIMARY_COLOR,
        ),
        borderRadius: BorderRadius.circular(8.0),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: IntrinsicHeight( // 높이를 내부 위젯들의 최대 높이로 설정
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _Time( // 시작과 종료 시간을 보여줄 위젯
                startTime: startTime, 
                endTime: endTime,
              ),
              SizedBox(width: 16.0),
              _Content( // 일정 내용을 보여줄 위젯
                content: content
              ),
              SizedBox(width: 16.0),
            ],
          ),
        ),
      ),
    );
  }
}

// 시간을 표현할 _Time 위젯
class _Time extends StatelessWidget {
  final int startTime; // 시작 시간 저장
  final int endTime; // 종료 시간 저장
  
  // 생성자, 외부로부터 시작/종료 시간을 받아온다
  const _Time({
    required this.startTime,
    required this.endTime,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    final textStyle = TextStyle(
      fontWeight: FontWeight.w600,
      color: PRIMARY_COLOR,
      fontSize: 16.0,
    );

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text( // 9 -> 09:00 처럼 두 자리로 맞춤
          "${startTime.toString().padLeft(2, '0')}:00",
          style: textStyle,
        ),
        Text(
          "${endTime.toString().padLeft(2, '0')}:00",
          style: textStyle.copyWith(
            fontSize: 10.0,
          ),
        )
      ],
    );
  }
}

// 내용을 렌더링할 _Content 위젯
class _Content extends StatelessWidget {
  final String content; // 내용 저장

  const _Content({
    required this.content, // 외부로 부터 content를 전달 받음
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    
    return Expanded( // 최대한 넓게 늘리기
      child: Text(
        content,
      )
    );
  }
}