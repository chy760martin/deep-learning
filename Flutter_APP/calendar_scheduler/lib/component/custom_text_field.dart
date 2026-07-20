import 'package:calendar_scheduler/const/colors.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class CustomTextField extends StatelessWidget {
  final String label; // 텍스트 필드 제목을 변수에 저장한다
  final bool isTime; // 시간 선택하는 텍스트 필드인지 여부
  final FormFieldSetter<String> onSaved;
  final FormFieldValidator<String> validator;

  const CustomTextField({
    required this.label, // 외부로부터 텍스트 필드 제목 받아온다
    required this.isTime, // 외부로부터 텍스트 필드 여부(true, false) 받아온다
    required this.onSaved, // 외부로부터 값 저장시 실행할 함수 받아온다
    required this.validator, // 외부로부터 갑 검증시 실행할 함수 받아온다
    super.key
  });

  @override
  Widget build(BuildContext context) {
    return Column( // 세로운 텍스트와 텍스트 필드 배치
      crossAxisAlignment: CrossAxisAlignment.start, // 왼쪽(가로 시작점)에 맞추어 정렬
      children: [
        Text(
          label,
          style: TextStyle(
            color: PRIMARY_COLOR,
            fontWeight: FontWeight.w600,
          ),
        ),
        Expanded(
          // flex: 0 시간 필드일때는 최소 크기만 차지, flex: 1 일반 텍스트 필드일때는 남는 공간을 확장해서 차지함
          flex: isTime ? 0 : 1,
          child: TextFormField(
            onSaved: onSaved, // 폼 저장했을때 실행할 함수
            validator: validator, // 폼 검증했을때 실행할 함수
            cursorColor: Colors.grey,
            // 1 필드는 한줄, 아니면 여러줄 가능
            maxLines: isTime ? 1 : null,
            expands: !isTime, // 일반 필드는 공간을 최대한 확장
            // 시간 관련 텍스트 필드는 기본 숫자 키보드, 아니면 일반 글자 키보드 보여주기
            keyboardType: isTime ? TextInputType.number : TextInputType.multiline,
            // 시간 관련 텍스트 필드는 숫자만 입력하도록 제한
            inputFormatters: isTime ? [
              FilteringTextInputFormatter.digitsOnly,
            ] : [],
            decoration: InputDecoration(
              border: InputBorder.none, // 테두리 삭제
              filled: true, // 배경색 지정 선언
              fillColor: Colors.grey[300],
              suffixText: isTime ? "시" : null, // 시간 관련 텍스트 필드는 "시" 접미사 추가
            ),
          ),
        ),
      ],
    );
  }
}