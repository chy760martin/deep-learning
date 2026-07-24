import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart'; // geolocator 패키기 임포트

class Loading extends StatefulWidget {
  const Loading({super.key});

  @override
  State<Loading> createState() => _LoadingState();
}

class _LoadingState extends State<Loading> {
  @override
  void initState() {
    super.initState();
    getLocation();
  }
  // 위도/경도 위치 정보
  void getLocation() async {
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