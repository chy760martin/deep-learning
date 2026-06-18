import 'package:flutter/material.dart';
import '../models/cat.dart'; // Cat 객체 사용

// 서버에서 댓글 불러오기 전에 사용할 임시 데이터
final List<String> replies = [
  "저 근엄한 눈빛",
  "어느 고양이별에서 왔니?",
  "집사로서 주인님께 충성할뿐...",
  "냥이님 날 가져요~~~",
  "왕족 고양이라서 '오히려 좋아'!",
  "중요한 건 꺾이지 않는 냥미모",
];

class DetailScreen extends StatefulWidget {
  const DetailScreen({
    super.key,
    required this.cat,
  });
  // Cat 객체를 외부로부터 받아서 사용
  final Cat cat;

  @override
  State<DetailScreen> createState() => _DetailScreenState();
}

// "_" private, "없을시" public
class _DetailScreenState extends State<DetailScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.cat.title), // 고양이의 제목을 앱바에 표시
      ),
      // Scaffold 내부에 body로 ListView()를 넣어 작성
      body: SafeArea( // SafeArea 로 안전 영역 확보 후 ListView 로 전체 내용을 스크롤 가능하게 만든다
        child: ListView(
          padding: const EdgeInsets.only(
            top: 10.0,
            left: 10.0,
            right: 10.0,
          ),
          physics: const ClampingScrollPhysics(),
          children: [
            AspectRatio(
              aspectRatio: 1, // 정사각형 비율
              child: Image.asset( // 고양이 이미지를 표시
                widget.cat.link, // 이미지 경로
                fit: BoxFit.cover,
              ),
            ),
            Row( // 왼쪽 고양이 이름 + 오른쪽 좋아요 버튼
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  widget.cat.name, // 고양이 이름
                  style: const TextStyle(
                    fontSize: 20.0,
                    color: Color(
                      0xFF777777,
                    ),
                  ),
                ),
                Row(
                  children: [
                    IconButton( // 좋아요 버튼
                      padding: EdgeInsets.zero,
                      icon: Icon(
                        Icons.thumb_up_outlined,
                      ),
                      onPressed: () {},
                    ),
                    Text( // 좋아요 카운트
                      widget.cat.likeCount.toString(),
                    ),
                  ],
                )
              ],
            ),
            Text( // 댓글 개수 표시
              "댓글 ${widget.cat.replyCount}개",
            ),
            // 임시 댓글 데이터를 반복해서 표시
            ...List.generate(
              replies.length, // 첫번째 인자: 리스트 길이(replies.length)
              (int index) => Padding( //두번째 인자: 각 인텍스별로 어떤 요소를 만들지 정의하는 함수
                padding: const EdgeInsets.only(
                  top: 10.0,
                ),
                child: Row(
                  children: [
                    const Text(
                      "익명",
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                    const Padding(padding: EdgeInsets.symmetric(horizontal: 3.0)),
                    Text(
                      replies[index],
                    ),
                  ],
                ),
              )
            ),
            Padding(
              padding: const EdgeInsets.only(
                top: 10.0,
              ),
              child: Text( // 고양이 등록 날짜를 표시
                "${widget.cat.created.year}년 ${widget.cat.created.month}월 ${widget.cat.created.day}일",
                style: const TextStyle(
                  color: Color(
                    0xFFAAAAAA,
                  )
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}