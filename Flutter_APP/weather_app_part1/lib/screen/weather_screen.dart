import 'package:flutter/material.dart';
import 'package:weather_app_part1/screen/weather_icon.dart'; // 날씨 아이콘

class WeatherScreen extends StatefulWidget {
  const WeatherScreen({
    this.parseWeatherData, // 생성자에 속성 추가
    super.key
  });

  final dynamic parseWeatherData; // 데이터를 전달 받을 속성

  @override
  State<WeatherScreen> createState() => _WeatherScreenState();
}

class _WeatherScreenState extends State<WeatherScreen> {
  String? cityName; // 도시명
  int? myTemp; // 온도
  Widget? icon; // 아이콘
  WeatherIcon weatherIcon = WeatherIcon(); // 아이콘 객체 생성

  @override
  void initState() {
    // 이 모든 과정은 WeatherScreen 위젯이 생성될때 이루어져야 하므로, initState() 메서드 안에서 updateData() 메서드를 호출해준다, 
    // 이때 updateData() 메서드의 인자 값으로 widget 속성을 사용해서 parseWeatherData 속성에 접근해서 값을 가져온다
    super.initState();
    updateData(widget.parseWeatherData);
  }

  // 매번 새로운 날씨 데이터를 전달받을 매개변수 추가
  void updateData(dynamic weatherData) {
    // 날씨 데이터 업데이트 로직 추가
    cityName = weatherData['name'];
    double myTemp2 = weatherData['main']['temp'];
    myTemp = myTemp2.round();

    // OpenWeatherMap 날씨 데이터 id 값 가져온다
    int condition = weatherData['weather'][0]['id'];
    icon = weatherIcon.getWeatherIcon(condition);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      body: SafeArea(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                "$cityName",
                style: const TextStyle(
                  fontSize: 35,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
              SizedBox(
                height: 20.0,
              ),
              Text(
                "$myTemp",
                style: TextStyle(
                  fontSize: 85,
                  fontWeight: FontWeight.w300,
                  color: Colors.white
                ),
              ),
              // icon != null: icon null 아니면 
              // ? icon as Widget: icon as Widget으로 캐스팅 반환
              // const SizedBox.shrink(): null 이면 빈 위젯을 반환
              icon != null ? icon as Widget : const SizedBox.shrink(),
            ],
          ),
        )
      ),
    );
  }
}