import 'package:calendar_scheduler/component/custom_text_field.dart';
import 'package:calendar_scheduler/const/colors.dart';
import 'package:flutter/material.dart';

class ScheduleBottomSheet extends StatefulWidget {
  const ScheduleBottomSheet({super.key});

  @override
  State<ScheduleBottomSheet> createState() => _ScheduleBottomSheetState();
}

class _ScheduleBottomSheetState extends State<ScheduleBottomSheet> {
  @override
  Widget build(BuildContext context) {
    // 키보드 높이 가져오기, 시스템이 차지하는 화면 아랫부분 크기를 알 수 있음
    final bottomInset = MediaQuery.of(context).viewInsets.bottom;

    return SafeArea(
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
                    ),
                  ),
                  const SizedBox(width: 16.0),
                  Expanded(
                    child: CustomTextField( // 종료 시간 입력 필드
                      label: "종료 시간", 
                      isTime: true
                    ),
                  ),
                ],
              ),
              SizedBox(height: 8.0),
              Expanded(
                child: CustomTextField( // 내용 입력 필드
                  label: "내용", 
                  isTime: false
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
    );
  }

  // onSavePressed 저장 버튼 함수
  void onSavePressed() {

  }
}