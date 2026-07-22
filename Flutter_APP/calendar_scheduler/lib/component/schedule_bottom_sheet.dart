import 'package:calendar_scheduler/component/custom_text_field.dart';
import 'package:calendar_scheduler/const/colors.dart';
import 'package:calendar_scheduler/database/drift_database.dart';
import 'package:drift/drift.dart' hide Column;
import 'package:flutter/material.dart';
import 'package:get_it/get_it.dart';

class ScheduleBottomSheet extends StatefulWidget {
  final DateTime selectedDate; // 선택된 날짜 상위 위젯에서 입력값을 저장

  const ScheduleBottomSheet({
    required this.selectedDate, // 외부로부터 선택된 날짜를 받아온다
    super.key,
  });

  @override
  State<ScheduleBottomSheet> createState() => _ScheduleBottomSheetState();
}

class _ScheduleBottomSheetState extends State<ScheduleBottomSheet> {
  final GlobalKey<FormState> formKey = GlobalKey(); // 폼 key 생성

  int? startTime; // 시작 시간 저장 변수
  int? endTime; // 종료 시간 저장 변수
  String? content; // 일정 내용 저장 변수

  @override
  Widget build(BuildContext context) {
    // 키보드 높이 가져오기, 시스템이 차지하는 화면 아랫부분 크기를 알 수 있음
    final bottomInset = MediaQuery.of(context).viewInsets.bottom;

    return Form( // 텍스트 필드를 한번에 관리 할 수 있는 폼
      key: formKey,
      child: SafeArea(
        child: Container(
          // ScheduleBottomSheet StatefulWidget 위젯을 생성하고 
          // SafeArea 위젯에 화면의 반을 차지하는 흰색 Container 위젯을 배치
          // 화면의 반 높이에 키보드 높이 추가
          height: MediaQuery.of(context).size.height / 2 + bottomInset,
          color: Colors.white,
          child: Padding(
            // 패딩에 키보드 높이 추가해서 위젯 전반적으로 위로 올려준다
            padding: EdgeInsets.only(left: 8, right: 8, top: 8, bottom: bottomInset),
            child: Column(
              children: [
                Row( // 시작/종료 시간 가로로 배치
                  children: [
                    Expanded(
                      child: CustomTextField( // 시작 시간 입력 필드
                        label: "시작 시간", 
                        isTime: true,
                        onSaved: (String? val) {
                          // 저장이 실행되면 startTime 변수에 텍스트 필드값 저장
                          startTime = int.parse(val!);
                        },
                        validator: timeValidator,
                      ),
                    ),
                    const SizedBox(width: 16.0),
                    Expanded(
                      child: CustomTextField( // 종료 시간 입력 필드
                        label: "종료 시간", 
                        isTime: true,
                        onSaved: (String? val) {
                          // 저장이 실행되면 endTime 변수에 텍스트 필드값 저장
                          endTime = int.parse(val!);
                        },
                        validator: timeValidator,
                      ),
                    ),
                  ],
                ),
                SizedBox(height: 8.0),
                Expanded(
                  child: CustomTextField( // 내용 입력 필드
                    label: "내용", 
                    isTime: false,
                    onSaved: (String? val) {
                      // 저장이 실행되면 content 변수에 텍스트 필드값 저장
                      content = val;
                    },
                    validator: contentValidator,
                  ),
                ),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton( // 저장 버튼 추가
                    onPressed: onSavePressed, 
                    style: ElevatedButton.styleFrom(
                      backgroundColor: PRIMARY_COLOR,
                    ),
                    child: Text("저장"),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // onSavePressed 저장 버튼 함수
  void onSavePressed() async {
    // 버튼 클릭시 -> validate() 실행 -> 모든 validator(timeValidator/contentValidator) 함수 호출
    if (formKey.currentState!.validate()) {
      // 검증 성공시 -> save() 실행 -> 모든 onSaved 함수 호출
      formKey.currentState!.save(); // 폼 저장하기
      
      // 일정 생성하기, LocalDatabase 클래스 createSchedule 함수 실행
      await GetIt.I<LocalDatabase>().createSchedule(
        // 단 매개변수에 SchedulesCompanion 클래스에는 실제 Schedules 테이블 입력될 값들을
        // 드리프트 패키지에서 제공하는 Value라는 클래스로 감싸서 입력해줘야 한다
        SchedulesCompanion(
          startTime: Value(startTime!),
          endTime: Value(endTime!),
          content: Value(content!),
          date: Value(widget.selectedDate),
        ),
      );

      Navigator.of(context).pop(); // 일정 생성 후 화면 뒤로 가기
    }
  }

  // 시간값 검증
  String? timeValidator(String? val) {
    if (val == null) {
      return '값을 입력해주세요';
    }

    int? number;

    try {
      number = int.parse(val);
    } catch (e) {
      return '숫자를 입력해주세요';
    }

    if (number < 0 || number > 24) {
      return '0시부터 24시 사이를 입력해주세요';
    }

    return null; // 에러없음, 즉 검증 성공을 의미한다
  }

  // 내용값 검증
  String? contentValidator(String? val) {
    if (val == null || val.length == 0) {
      return '값을 입력해주세요';
    }

    return null;
  }
}