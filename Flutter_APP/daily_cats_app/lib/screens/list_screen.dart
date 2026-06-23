import 'package:daily_cats_app/screens/detail_screen.dart';
import 'package:daily_cats_app/screens/upload_screen.dart';
import 'package:daily_cats_app/screens/register_screen.dart';
import 'package:flutter/material.dart';
import '../models/cat.dart';

final List<Cat> cats = [
  Cat(
    id: "0", 
    name: "별님이", 
    title: "오늘의 귀염둥이", 
    link: "assets/images/cat_00.jpg", 
    likeCount: 1930, 
    replyCount: 6, 
    created: DateTime(2022, 11, 13, 22, 14, 53, 982),
  ),
  Cat(
    id: "1", 
    name: "버찌", 
    title: "너만 본단 말이야", 
    link: "assets/images/cat_01.jpg", 
    likeCount: 3023, 
    replyCount: 9, 
    created: DateTime(2022, 10, 24, 11, 00, 23, 689),
  ),
  Cat(
    id: "2", 
    name: "레이", 
    title: "암 소 씨리어스", 
    link: "assets/images/cat_02.jpg", 
    likeCount: 1003, 
    replyCount: 2, 
    created: DateTime(2022, 1, 6, 11, 14, 9, 353),
  ),
  Cat(
    id: "3", 
    name: "굿보이", 
    title: "고양이와 함께 춤을", 
    link: "assets/images/cat_03.jpg", 
    likeCount: 2012, 
    replyCount: 53, 
    created: DateTime(2021, 12, 31, 23, 59, 59, 999),
  ),
  Cat(
    id: "4", 
    name: "차라", 
    title: "이래뵈어도 난 왕족 고양이야", 
    link: "assets/images/cat_04.jpg", 
    likeCount: 443, 
    replyCount: 1, 
    created: DateTime(2022, 4, 23, 17, 32, 50, 725),
  ),
];

class ListScreen extends StatefulWidget {
  const ListScreen({super.key});

  @override
  State<ListScreen> createState() => _ListScreenState();
}

class _ListScreenState extends State<ListScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Daily Cats"),
        actions: [
          // 업로드 화면으로 이동
          IconButton( // 카메라 버튼 클릭시 "사진 업로드" 다이얼로그가 나오도록 함
            icon: const Icon(Icons.camera_alt),
            onPressed: () {
              showDialog(
                context: context, 
                builder: (_) => const UploadScreen(), // (_) (context) 매개변수 사용 안함, 사진 업로드 화면
              );
            },
          ),
          // 회원가입 화면으로 이동
          IconButton(
            icon: const Icon(Icons.person_add), // 회원가입 아이콘
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(
                  builder: (context) => const RegisterScreen(), // 회원가입 화면으로 이동
                ),
              );
            },
          ),
        ],
      ),
      body: GridView.builder(
        padding: const EdgeInsets.only(
          top: 10.0,
          left: 10.0,
          right: 10.0,
        ),
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 3, // 고양이 사진이 한줄에 3개씩 표시
          mainAxisSpacing: 10.0,
          crossAxisSpacing: 10.0,
          childAspectRatio: 1.0, // 너비와 높이가 동일한 정사각형 모양으로 사진을 보여줌
        ),
        itemCount: cats.length, 
        itemBuilder: (_, int index) => GestureDetector(
          // 상세 화면으로 이동
          // onTap() 이벤트가 발생하면 리스트 화면 위에 상세 화면을 쌓아 올린다(실제 사용자는 상세 화면만 보임)
          onTap: () {
            // 현재 화면(context)에서 새로운 화면(Route)을 스택(stack)에 추가(push), 닫기(pop)
            Navigator.of(context).push( // Navigator.of(context).push() 메서드를 사용했을 경우, 앱바[뒤로가기] 버튼 및 동작은 자동 구현된다
              MaterialPageRoute( // 화면 전환 클래스, 머티리얼 디자인 스타일의 화면 전환 애니메이션 제공
                // 새로운 화면을 어떻게 그릴지 정의하는 부분이고, 여기서는 DetailScreen 위젝을 생성해서 보여준다
                builder: (context) => DetailScreen(
                  cat: cats[index], // 현재 리스트에서 선택한 고양이 데이터를 DetailScreen에 전달
                  ),
                ),
            );
          },
          child: Image.asset(
            cats[index].link,
            fit: BoxFit.cover,
          ),
        )
      ),
    );
  }
}