import 'package:http/http.dart' as http;
import 'dart:convert';

class Network {
  Network(this.url); // url 전달
  final String url;

  // Future<dynamic> String,int,double  여러가지 타입 
  Future<dynamic> getJsonData() async {
    // Json 데이터 파싱 구현
    http.Response response = await http.get(Uri.parse(url)); // 더미 데이터 링크를 url 변수로 대체

    if (response.statusCode == 200) {
      String jsonData = response.body;
      var parsingData = jsonDecode(jsonData);
      return parsingData; // json 데이터 반환
    }
  }
}