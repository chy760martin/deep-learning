// 회원 가입 입력,폼 사용자의 입력을 받고 사용자와 상호 작용하는 위젯, 주로 상태 관리 및 서버 호출과 연관됨

import 'package:flutter/material.dart';


class RegisterScreen extends StatefulWidget {
  const RegisterScreen({Key? key,}) : super(key: key);

  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  bool _agreed = false; // 약관 동의 상태 변수

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar( // appbar 제목
        title: const Text("회원가입"),
      ),
      body: ListView( // body 영역
        padding: const EdgeInsets.symmetric(
          vertical: 20.0,
          horizontal: 30.0
        ),
        children: [
          const Text(
            "다음 정보를 모두 입력해 주세요.",
            textAlign: TextAlign.center,
          ),
          const TextField(
            autocorrect: false, // 맞춤법,철자 자동 교정, 이름에서는 불필요
            autofocus: true, // 화면이 열리자마자 해당 입력창에 커서가 자동으로 위치
            decoration: InputDecoration( // 입력창의 외형을 꾸미는 속성
              hintText: "이름" // 입력창이 비어 있을때 표시되는 문구
            ),
            keyboardType: TextInputType.name, // 이름 입력에 적합한 키보드가 표시
            textInputAction: TextInputAction.next, // 키보드의 Enter 버튼이 다음으로 표시 -> 누르면 다음 입력간으로 포커스 이동
          ),
          const TextField(
            autocorrect: false,
            decoration: InputDecoration(
              hintText: "이메일"
            ),
            keyboardType: TextInputType.emailAddress,
            textInputAction: TextInputAction.next,
          ),
          const TextField(
            autocorrect: false,
            decoration: InputDecoration(
              hintText: "비밀번호",
            ),
            textInputAction: TextInputAction.next,
            obscureText: true, // 비밀번호 입력 스타일 true
          ),
          const TextField(
            autocorrect: false,
            decoration: InputDecoration(
              hintText: "비밀번호 확인"
            ),
            textInputAction: TextInputAction.done,
            obscureText: true,
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Switch(
                value: _agreed, 
                onChanged: (bool newValue) {
                  setState(() { // 토클시 상태를 업데이트
                    _agreed = newValue;
                  });
                },
              ),
              const Text("이용약관에 동의합니다."),
            ],
          ),
          ElevatedButton.icon(
            icon: const Icon(Icons.send), 
            label: const Text("제출"),
            onPressed: _agreed 
              ? () {
                // 젳출 로직
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text("제출 완료!"))
                );
              }
              : null, // 약관 동의하지 않으면 버튼 비활성화
          ),
        ],
      ),
    ) ;
  }
}