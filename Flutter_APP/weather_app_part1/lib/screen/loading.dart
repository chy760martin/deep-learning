import 'package:flutter/material.dart';
// import 'package:geolocator/geolocator.dart'; // geolocator 패키지 임포트
// import 'package:http/http.dart' as http; // http 패키지 임포트 http 별칭 사용
// import 'dart:convert';
import 'package:weather_app_part1/data/my_location.dart';
import 'package:weather_app_part1/data/network.dart';
import 'package:weather_app_part1/screen/weather_screen.dart'; // JSON 데이터를 파싱 사용
const String apiKey = ''; // OpenWeather Map 키 값

class Loading extends StatefulWidget {
  const Loading({super.key});

  @override
  State<Loading> createState() => _LoadingState();
}

class _LoadingState extends State<Loading> {
  double? myLongitude2; // 경도
  double? myLatitude2; // 위도

  @override
  // 앱 초기 실행
  void initState() {
    super.initState();
    getLocation();
  }
  // 위도/경도 위치 정보
  void getLocation() async {
    MyLocation myLocation = MyLocation();
    await myLocation.getMyCurrentLocation();
    myLatitude2 = myLocation.myLatitude; // 위도
    myLongitude2 = myLocation.myLongitude; // 경도

    // network 인스턴스 생성하고 생성자에 더미 데이터 url 전달
    // 좌표기반 주소, 위도/경도
    // Network network = Network('https://api.openweathermap.org/data/2.5/weather?lat=${myLatitude2}&lon=${myLongitude2}&appid=${apiKey}');
    // 도시 이름 기반 주소, 화씨->섭씨
    Network network = Network('https://api.openweathermap.org/data/2.5/weather?q=Seoul&appid=${apiKey}&units=metric');
    var weatherData = await network.getJsonData(); // Json 데이터 변환
    print(weatherData);
    // async 메서드가 비동기 작업을 수행하는 동안 해당 State 객체가 위쳇 트리에서 해제(!mounted) 되었는지 확인하고,
    // 만약 해제되었다면 setState() 메서드를 호출하지 않도록 한다
    if (!mounted) return;
    Navigator.push(context, MaterialPageRoute(
      builder: (context) => WeatherScreen(parseWeatherData: weatherData,)
    ));
  }

  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      body: Center(
        child: FilledButton(
          onPressed: () {
            getLocation();
          }, 
          child: const Text('Get location')
        ),
      ),
    );
  }
}