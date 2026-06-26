import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class ApiScreen extends StatefulWidget {
  const ApiScreen({super.key});

  @override
  State<ApiScreen> createState() => _ApiScreenState();
}

class _ApiScreenState extends State<ApiScreen> {
  // API 호출 예시
  String message = "Loading...";

  @override
  void initState() {
    // 부모 클래스의 초기화 로직 실행
    super.initState();
    getData(); // 앱 시작시 자동으로 API 호출
  }

  // Future<void> getData() async -> 비동기 함수 선언, await를 사용해 네트워크 요청이 끝날때까지 기다릴수 있음, 반환값은 특별히 없으므로 void
  Future<void> getData() async {
    // Uri.https("catfact.ninja", "fact") -> HTTPS 프로토콜로 https://catfact.ninja/fact 주소를 만든다
    final Uri uri = Uri.https("catfact.ninja", "fact");
    // await http.get(uri) -> API 서버에 GET 요청을 보냄, 응답이 올때까지 기다린 뒤 response 에 저장
    final http.Response response = await http.get(uri);

    // 정상 응답 여부 체크
    if (response.statusCode == 200) {
      // jsonDecode(response.body) 응답 본문을 JSON 으로 반환
      final Map<String, dynamic> body = jsonDecode(response.body);

      // setState()로 상태 갱신, UI가 다시 빌드되어 화면에 새로운 데이터 표시
      setState(() {
        // body["fact"] -> 가져온 데이터를 message 변수에 저장
        message = body["fact"];
      });
    } else {
      setState(() {
        message = "에러: ${response.statusCode}";
      });
    }
  }

  // 실제 화면 구성
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Cat Facts"),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(message,),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: getData,
        tooltip: "Increment",
        child: const Icon(Icons.refresh),
      ),
    ) ;
  }
}