// models/cat.dart

// 클래스 내부에 들어갈 필드의 이름과 타입을 정의
class Cat {
  final String id; // 고양이 게시물의 고유 ID, 추후 파이어베이스에서 자동 생성
  final String name; // 고양이 이름
  final String title; // 고양이 게시물 제목
  final String link; // 고양이 사진 링크,URL
  final int likeCount; // '좋아요' 수
  final int replyCount; // '댓글' 수
  final DateTime created; // 게시물 생성 시각(년,월,시,분,초,밀리초)

  // final 선언된 변수는 변할 수 없음, 이 경우 무조건 초기값이 존재해야 하므로 required 키워드가 필요함.
  //  필드 값들을 받아 클래스의 인스턴스를 생성하는 생성자 함수
  Cat({
    required this.id,
    required this.name,
    required this.title,
    required this.link,
    required this.likeCount,
    required this.replyCount,
    required this.created,
  });
}