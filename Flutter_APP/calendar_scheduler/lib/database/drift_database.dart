// drift 생성 코드 및 아래와 같이 명령어 실행
// 터미널에서 실행 명령어 : flutter pub run build_runner build
import 'package:calendar_scheduler/model/schedule.dart';
import 'package:drift/drift.dart';

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

}