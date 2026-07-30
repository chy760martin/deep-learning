// REST API Schedules 모델 구현
class ScheduleModel {
  // 외부로 부터 받아온 값을 변수에 저장
  final String id;
  final String content;
  final DateTime date;
  final int startTime;
  final int endTime;

  // 외부로 부터 받아 오는 값
  ScheduleModel({
    required this.id,
    required this.content,
    required this.date,
    required this.startTime,
    required this.endTime,
  });

  // JSON 형식의 데이터 -> ScheduleModel 모델에 매핑
  ScheduleModel.fromJson({
    // 외부로부터 받아오는 값: 타입은 String, dynamic
    required Map<String, dynamic> json,
  }) : id = json['id'],
      content = json['content'],
      date = DateTime.parse(json['date']),
      startTime = json['startTime'],
      endTime = json['endTime'];
  
  // ScheduleModel 모델 -> JSON 변환
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'content': content,
      'date': '${date.year}${date.month.toString().padLeft(2, '0')}${date.day.toString().padLeft(2, '0')}',
      'startTime': startTime,
      'endTime': endTime,
    };
  }

  // 현재 모델을 특정 속성만 변환해서 새로 생성
  ScheduleModel copyWith({
    String? id,
    String? content,
    DateTime? date,
    int? startTime,
    int? endTime,
  }) {
    return ScheduleModel(
      // 리턴값 설정: 값이 있으면 변경하고, 없거나 null 이면 기존 값을 유지
      id: id ?? this.id,
      content: content ?? this.content, 
      date: date ?? this.date, 
      startTime: startTime ?? this.startTime, 
      endTime: endTime ?? this.endTime,
    );
  }
}