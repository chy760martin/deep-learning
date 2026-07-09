import 'package:flutter/material.dart';

// 쿠퍼티노 IOS 위젯 사용하기 위함
import 'package:flutter/cupertino.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  DateTime firstDay = DateTime.now();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.pink[100], // 배경색을 연한 핑크색으로 설정
      body: SafeArea( // 시스템 UI 영역을 침범하지 않도록 SafeArea 위젯으로 감싸줌
        top: true, // 상단 영역은 침범하지 않도록 설정
        bottom: false, // 하단 영역은 침범하도록 설정
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceBetween, // 위젯들을 위와 아래로 배치
          crossAxisAlignment: CrossAxisAlignment.stretch, // 위젯들을 가로로 꽉 채우도록 설정
          children: [
            _DDay(
              // 하트 버튼 클릭 시 동작할 코드 작성
              onHeartPressed: onHeartPressed,
              firstDay: firstDay, // firstDay를 _DDay 위젯에 전달
            ),
            _CoupleImage(),
          ],
        ),
      ),
    );
  }
  // 하트 버튼 클릭 시 동작할 함수
  void onHeartPressed() {
    // 쿠퍼티노 다이럴로그 실행
    showCupertinoDialog(
      context: context,
      builder: (BuildContext context) {
        return Align( // 정렬 위젯
          alignment: Alignment.bottomCenter,
          child: Container(
            color: Colors.white,
            height: 300,
            child: CupertinoDatePicker(
              mode: CupertinoDatePickerMode.date, // 날짜만 선택할 수 있도록 설정
              onDateTimeChanged: (DateTime date) { // 사용자가 날짜를 바꾸면 호출되는 콜백
                setState(() { // 상태를 갱신하면 화면이 다시 그려지고 _DDay 클래스 위젯에 업데이트
                  firstDay = date;
                });
              },
            )
          ),
        );
      },
      barrierDismissible: true // 외부 탭할 경우 다이얼로그 닫기
    );
  }
}


class _DDay extends StatelessWidget {
  // 하트 눌렀을때 실행할 함수
  final GestureTapCallback onHeartPressed;
  final DateTime firstDay;

  _DDay({
    // 상위에서 함수 입력받기
    required this.onHeartPressed,
    required this.firstDay,
  });

  @override
  Widget build(BuildContext context) {
    // 테마 블러오기
    final textTheme = Theme.of(context).textTheme;
    // 현재 날짜 시간
    final now = DateTime.now();

    return Column(
      children: [
        const SizedBox(height: 16.0), // 상단 여백
        Text(
          'U&I',
          style: textTheme.headlineLarge, // 테마에서 정의한 displayLarge 스타일 적용
        ),
        const SizedBox(height: 16.0), // 텍스트와 날짜 사이의 여백
        Text(
          '우리 처음 만난 날',
          style: textTheme.titleLarge,
        ),
        Text(
          // 날짜를 년.월.일 형식으로 표시
          '${firstDay.year}.${firstDay.month}.${firstDay.day}',
          style: textTheme.bodyMedium,
        ),
        const SizedBox(height: 16.0),
        IconButton(
          iconSize: 60.0,
          onPressed: onHeartPressed, // 아이콘 눌렀을때 실행할 함수
          icon: Icon(
            Icons.favorite,
            color: Colors.red,
          ),
        ),
        const SizedBox(height: 16.0),
        Text(
          // D-Day 계산: 오늘 날짜 DateTime(now.year, now.month, now.day),
          // 오늘 날짜 - firstDay를 계산 difference(firstDay),
          // Duration 객체를 일(day)단위로 변환 + 1 .inDays 
          'D+${DateTime(now.year, now.month, now.day).difference(firstDay).inDays + 1}',
          style: textTheme.bodyMedium,
        ),
      ],
    );
  }
}

class _CoupleImage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    // Expanded 위젯을 사용하여 자동 공간 채우기
    return Expanded(
      child: Center(
          child: Image.asset(
          'assets/img/middle_image.png', // 이미지 경로를 지정

          // Expanded가 우선순위를 갖게 되어 무시된다.
          height: MediaQuery.of(context).size.height / 2,
        ),
      ),
    );
  }
}