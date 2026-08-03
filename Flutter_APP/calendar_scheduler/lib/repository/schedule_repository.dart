// API 요청 기능 구현
import 'dart:async'; // 비동기 프로그램밍 지원 라이브러리, Future/Stream 타입, 서버에서 데이터 받아올때
import 'dart:io'; // 파일/디렉토리/소켓 HTTP등 입출력(I/O) 기능 제공, 파일 읽기/쓰기 네트워크 요청 서버 소켓 열기

import 'package:calendar_scheduler/model/schedule_model.dart'; // Schedules 모델
import 'package:dio/dio.dart'; // HTTP 클라이언트 라이브러리, GET/POST/PUT/DELETE HTTP 요청

class ScheduleRepository {
  // HTTP 요청을 보내기 위한 클라이언트, Future 기반으로 동작
  final _dio = Dio();
  // iOS -> 127.0.0.1 (시뮬레이터에서 호스트 머신 접근시 필요), Android -> 10.0.2.2 (에뮬레이터에서 호스트 머신 접근 시 필요)
  final _targetUrl = 'http://${Platform.isIOS ? '127.0.0.1' : 'localhost'}:3000/schedule';

  // API GET 요청
  Future<List<ScheduleModel>> getSchedules({
    required DateTime date, // 외부로부터 날짜 데이터 받음
  }) async { // 비동기 요청
    final resp = await _dio.get( // await 서버 응답을 기다림
      _targetUrl,
      queryParameters: { // queryParameters GET 요청시 URL뒤에 붙는 ? date=20260730 쿼리 문자열을 만든다
        'date': '${date.year}${date.month.toString().padLeft(2, '0')}${date.day.toString().padLeft(2, '0')}',
      },
    );
    // JSON -> 모델 반환, 서버에서 받은 JSON 배열을 ScheduleModel 리스트로 반환
    // (resp.data as List) 서버 응답(JSON 배열)을 Dart 리스트로 캐스팅
    // .map<ScheduleModel>() 리스트의 각 요소를 반환, x는 JSON 객체 하나
    return (resp.data as List).map<ScheduleModel>(
      (x) => ScheduleModel.fromJson(json: x)
    ).toList();
  }

  // API POST 요청
  Future<String> createSchedule({
    required ScheduleModel schedule,
  }) async { // async{} 블록을 사용해서 API 요청
    final json = schedule.toJson(); // JSON 변환
    final resp = await _dio.post( // API POST 요청
      _targetUrl,
      data: json
    );
    return resp.data?['id']; // 응답에서 id 추출 후 반환, 최종 반환 타입은 Future<String>
  }

  // API DELETE 요청
  Future<String> deleteSchedule({
    required String id,
  }) async {
    final resp = await _dio.delete(
      _targetUrl,
      data: {
        'id': id, // 삭제할 id값
      }
    );
    return resp.data?['id']; // 삭제된 id값 반환
  }
}