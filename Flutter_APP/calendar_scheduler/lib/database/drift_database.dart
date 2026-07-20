// drift 생성 코드 및 아래와 같이 명령어 실행
// 터미널에서 실행 명령어 : flutter pub run build_runner build
import 'package:calendar_scheduler/model/schedule.dart';
import 'package:drift/drift.dart';

import 'package:drift/native.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'dart:io';

// private 값까지 불러올 수 있음
part 'drift_database.g.dart'; // part 파일 지정, Drift가 자동으로 생성하는 파일을 연결

// 사용할 테이블 등록, Schedules 테이블을 데이터베이스에 등록
@DriftDatabase(
  tables: [
    Schedules, // Schedules 클래스와 매핑
  ],
)

// 실제 앱에서 사용할 데이터베이스 클래스
class LocalDatabase extends _$LocalDatabase {
  // 1. LocalDatabase() 호출하면 -> _openConnection() 실행
  // 2. _openConnection()는 LazyDatabase 를 반환
  // 3. 이 반환값이 부모 클래스 _$LocalDatabase(QueryExecutor e)의 e 파라미터로 전달됨
  // 4. 부모 클래스는 이 QueryExecutor 를 기반으로 DB를 초기화
  LocalDatabase() : super(_openConnection());

  // select, 일정 조회 watchSchedules 커스텀 메서드: 특정 날짜의 일정들을 스트림으로 반환
  Stream<List<Schedule>> watchSchedules(DateTime date) =>
    // select(schedules): Drift가 자동으로 생성한 schedules 테이블에 대해 SELECT 쿼리 시작
    // ..where((tbl) => tbl.date.equals(date)): tbl은 schedules 테이블의 컬럼 객체, date를 비교 조건을 걸어준다
    // 쿼리 설명: 데이터를 조회하고 변화 감지(watch())
    (select(schedules)..where((tbl) => tbl.date.equals(date))).watch();

  // insert, 일정 등록 createSchedule 메서드에서 SchedulesCompanion 클래스 통해서 값을 넣어준다
  // Future<int> 비동기적으로 실행되며 완료시 int값을 반환, 삽입된 행의 기본 키(id)값
  // SchedulesCompanion 은 Schedules 테이블에 맞춰 자동 생성된 클래스
  Future<int> createSchedule(SchedulesCompanion data) =>
    into(schedules).insert(data);
  
  // delete, 일정 삭제
  Future<int> removeSchedule(int id) =>
    // delete() 함수에는 go() 함수를 실행해야 삭제가 완료됨
    (delete(schedules)..where((tbl) => tbl.id.equals(id))).go();
  
  // 드리프트 데이터베이스 클래스는 필수로 schemaVersion값을 지정해줘야 한다
  // 기본적으로 1부터 시작하고 테이블의 변화가 있을때마다 1씩 올려줘서 테이블 구조가 변경된다는 걸 드리프트에 인지시켜주는 기능
  @override
  int get schemaVersion => 1;
}

// 데이터베이스 연동, 드리프트 데이터베이스 객체는 부모 생성자에 LazyDatabase를 필수로 넣어줘야 한다
// LazyDatabase 객체에는 데이터베이스 생성할 위치에 대한 정보를 입력해주면 된다
LazyDatabase _openConnection() {
  return LazyDatabase(() async {
    // 데이터베이스 파일 저장할 폴더
    final dbFolder = await getApplicationDocumentsDirectory();
    final file = File(p.join(dbFolder.path, 'db_sqlite'));
    return NativeDatabase(file); // SQLite 데이터베이스 연결, Drift가 이 파일을 통해 데이터를 읽고/쓰기
  });
}