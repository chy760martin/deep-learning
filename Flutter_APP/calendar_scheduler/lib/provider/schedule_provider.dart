// 글로벌 상태 관리 구현
import 'package:calendar_scheduler/model/schedule_model.dart'; // ScheduleModel 모델
import 'package:calendar_scheduler/repository/schedule_repository.dart'; // API 요청

import 'package:flutter/material.dart';
// import 'package:provider/provider.dart'; // 글로벌 상태 관리 라이브러리, 앱 전체에서 일정 데이터 공유 및 변경시 UI가 자동으로 갱신

class ScheduleProvider extends ChangeNotifier {
  final ScheduleRepository repository; // API 요청 로직을 담은 클래스

  DateTime selectedDate = DateTime.utc( // 현재 날짜를 기본 선택값으로 설정
    DateTime.now().year,
    DateTime.now().month,
    DateTime.now().day,
  );

  // 일정 정보를 캐시 변수에 저장
  Map<DateTime, List<ScheduleModel>> cache = {}; // 날짜별 일정 캐시

  // ScheduleProvider 생성자, 초기화 시 현재 날짜 일정 불러오기
  ScheduleProvider({
    required this.repository, // 외부로부터 repository 값을 받아옴
  }) : super() {
    getSchedules(date: selectedDate);
  }

  // 일정 불러오기
  void getSchedules({
    required DateTime date, // 외부로부터 날짜 가져옴
  }) async {
    // GET 메서드 보내기, 파라미터 date
    final resp = await repository.getSchedules(date: date);

    // 캐시에 저장, 선택한 날짜의 일정들 업데이트하기
    // 키값(date), 이미 해당 키가 존재할때 (oldValue) => newValue, 해당 키가 없을때 ifAbsent: () => resp
    cache.update(date, (value) => resp, ifAbsent: () => resp);

    // UI 갱신, 리슨하는 위젯들 업데이트하기, ChangeNotifierProvider{}로 상태를 주입받고 있는 위젯 트리 안에 포함된 경우
    notifyListeners();
  }

  // 일정 생성하기
  void createSchedule({
    required ScheduleModel schedule
  }) async {
    // ScheduleModel 모델의 date를 가져옴
    final targetDate = schedule.date;
    // POST 메서드 보내기, 파라미터 ScheduleModel 모델
    final saveSchedule = await repository.createSchedule(schedule: schedule);

    // 캐시 저장
    cache.update( // 날짜에 대한 키값이 존재하면 업데이트
      targetDate, 
      (value) => [
        ...value, // 기존 데이터 List<ScheduleModel>를 복사
        schedule.copyWith( // 새 일정 추가
          id: saveSchedule, // 서버에 받은 id 반영
        ),
      ]..sort( // 시작 시간 기준 정렬
        (a, b) => a.startTime.compareTo(b.startTime),
      ),
      // 날짜에 해당되는 값이 없다면 새로운 리스트에 새로운 일정 하나만 추가
      ifAbsent: () => [schedule]
    );

    // UI 갱신, Provider를 구독하는 위젯들에게 상태 변경을 알린다
    notifyListeners();
  }

  // 일정 삭제하기
  void deleteSchedule({
    required DateTime date,
    required String id,
  }) async {
    // DELETE 메서드 보내기, 파리미터 String id
    final resp = await repository.deleteSchedule(id: id);

    // 삭제 id 확인
    if(resp == id) {
      cache.update( // 캐시에서 데이터 삭제
        date,
        // 기존 리스트(value)에서 id가 일치하지 않는 일정만 남긴다
        // 즉 삭제 대상 일정은 필터링되어 제거된다
        (value) => value.where((e) => e.id != id).toList(),
        // 만약 해당 날짜 키가 캐시에 없으면 빈 리스트를 반환한다
        ifAbsent: () => [],
      );

      // UI 갱신, Provider를 구독하는 위젯들에게 상태 변경을 알린다
      notifyListeners();
    } else {
      print("삭제 실패");
    }
  }

  // 날짜 변경
  void changeSelectedDate({
    required DateTime date,
  }) {
    // 현재 선택된 날짜를 매개변수로 입력받은 날짜로 변경
    selectedDate = date;
    // UI 갱신, Provider를 구독하는 위젯들에게 상태 변경을 알린다
    notifyListeners();
  }
}