// 데이터베이스 모델
import 'package:drift/drift.dart';

class Schedules extends Table {
  // IntColumn 정수형 컬럼, get if 테이블 id 컬럼, integer() 컬럼 타입을 INTEGER로 지정, autoIncrement() 자동증가 PRIMARY KEY 지정
  IntColumn get id => integer().autoIncrement()(); // PRIMARY KEY
  TextColumn get content => text()(); // 내용
  DateTimeColumn get date => dateTime()(); // 일정 날짜
  IntColumn get startTime => integer()(); // 시작 시간
  IntColumn get endTime => integer()(); // 종료 시간
}