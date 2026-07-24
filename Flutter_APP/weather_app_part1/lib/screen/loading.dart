import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart'; // geolocator 패키지 임포트
import 'package:http/http.dart' as http; // http 패키지 임포트 http 별칭 사용

class Loading extends StatefulWidget {
  const Loading({super.key});

  @override
  State<Loading> createState() => _LoadingState();
}

class _LoadingState extends State<Loading> {
  @override
  // 앱 초기 실행
  void initState() {
    print('test1');
    super.initState();
    getLocation();
    fetchData();
    print('test2');
  }
  // 위도/경도 위치 정보
  void getLocation() async {
    print('test3');
    // try/catch 예외 상활 처리
    try {
      // 권한 요청
      LocationPermission permission = await Geolocator.requestPermission();
      // 권한 체크
      if (permission == LocationPermission.denied) {
        return;
      }
      // 현재 위치
      Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high
      );
      print(position);
    } catch (e) {
      print('위치 정보수신에 문제가 생겼습니다 : ${e.toString()}');
    }
  }
  // 외부 인터넷에서 날씨 데이터를 가져오는 로직
  Future<void> fetchData() async { // 함수에 async 키워드를 붙여야 await를 사용 할 수 있음
    print('test4');
    // http.get()의 반환값(Future<Response>)을 await로 풀어서 Response 타입 변수에 담는다, http.get() 비동기 함수라서 Future<Response>를 반환
    final http.Response response = await http.get( // http.get() 메서드에 url 전달함(Uri.parse()를 통해서 url 정보)
      Uri.parse('https://api.openweathermap.org/data/2.5/weather?q=London&appid=')
    );

    // 정상 체크
    if (response.statusCode == 200) {
      print(response.body);
      print(response.statusCode);
    } else {
      print('Failed to load weather data');
    }
  }

  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      body: Center(
        child: FilledButton(
          onPressed: () {
            getLocation();
            fetchData();
          }, 
          child: const Text('Get location')
        ),
      ),
    );
  }
}