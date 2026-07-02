import 'package:flutter/material.dart'; // Flutter 기본 UI 위젯들을 사용하기 위해
import 'package:flutter/services.dart'; // 상태바 색상 변경을 위해
import 'dart:async'; // Timer를 사용하기 위해

// 상태 변화에 따라 UI를 다시 그려야 하는 위젯은 StatefulWidget을 상속받아야 함
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

// StatefulWidget을 상속받은 위젯의 상태를 관리하는 클래스
class _HomeScreenState extends State<HomeScreen> {
  // PageView를 제어하기 위한 PageController 생성
  final PageController pageController = PageController();

  // initState()는 위젯이 생성될 때 한 번만 호출되는 메서드 함수 등록
  // 사용시 앱을 새로 실행해야 적용 할 수 있음. (hot reload로는 적용되지 않음)
  @override
  void initState() {
    super.initState(); // 부모 initState() 호출

    // 3초마다 페이지를 전환하는 타이머 설정
    Timer.periodic(
      Duration(seconds: 3), // 3초마다
      (timer) {
        // 현재 페이지 인덱스 가져오기, null이면 0으로 초기화, 
        // int?는 null 허용 타입, pageController.page는 double 타입이므로 toInt()로 변환
        int? nextPage = pageController.page?.toInt() ?? 0;

        if (nextPage == 4) {
          nextPage = 0; // 마지막 페이지이면 첫 번째 페이지로 이동
        } else {
          nextPage++; // 다음 페이지로 이동
        }
        
        // 지정한 페이지로 애니메이션을 통해 이동
        pageController.animateToPage(
          nextPage,
          duration: Duration(milliseconds: 500), // 500ms 동안 애니메이션
          curve: Curves.easeInOut, // 애니메이션 곡선
        );
      }
    );
  }

  @override
  Widget build(BuildContext context) {

    // 상태바 색상 변경, 상태바가 이미 흰색이면 light 대신 dark를 주어 검정으로 변경
    SystemChrome.setSystemUIOverlayStyle(SystemUiOverlayStyle.light);

    return Scaffold(
      // 상단에 제목을 표시
      // appBar: AppBar(
      //   title: const Text('Image Carousel'),
      // ),
      // 실제 콘텐츠 표시
      body: PageView( // 여러 페이지(위젯)을 좌우로 스와이프하여 전환할 수 있는 위젯
        controller: pageController, // PageView를 제어할 PageController 전달

        // children: 스와이프할때 보여줄 위젯들의 리스트
        // 지금은 [1, 2, 3, 4, 5] 라는 숫자 리스트를 map() 함수를 이용하여 
        // Image.asset() 위젯으로 변환한 후 toList()로 리스트로 만들어 children에 전달
        children: [1, 2, 3, 4, 5].map(
          (number) => Image.asset(
            "assets/img/image_$number.jpeg",
            fit: BoxFit.cover, // 이미지가 위젯의 크기에 맞게 잘리거나 늘어나도록 설정
          ),
        )
        .toList(), // map() 함수는 Iterable을 반환하므로 toList()로 리스트로 변환
      ),
    );
  }
}