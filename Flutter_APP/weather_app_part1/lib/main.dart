import 'package:flutter/material.dart';
import 'package:weather_app_part1/screen/loading.dart';
// import 'package:geolocator/geolocator.dart';

void main() {
  runApp(const MyApp());
}


class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: Colors.blue,
      ),
      home: const Loading(), // loading.dart 파일의 Loading 위젯 호출
    );
  }

  // @override
  // Widget build(BuildContext context) {
  //   return MaterialApp(
  //     home: Scaffold(
  //       appBar: AppBar(title: const Text('Geolocator Test')),
  //       body: ElevatedButton(
  //         onPressed: () async {
  //           await getLocation(); // 여기서 호출
  //         }, 
  //         child: const Text('현재 위치 가져오기')
  //       ),
  //     ),
  //   );
  // }
}

// Future<void> getLocation() async {
//   // 권한 요청
//   LocationPermission permission = await Geolocator.requestPermission();

//   // 권한 체크
//   if (permission == LocationPermission.denied) {
//     return;
//   }

//   // 현재 위치
//   Position position = await Geolocator.getCurrentPosition(
//     desiredAccuracy: LocationAccuracy.high,
//   );

//   print('현재 위치: ${position.latitude}, ${position.longitude}');
// }