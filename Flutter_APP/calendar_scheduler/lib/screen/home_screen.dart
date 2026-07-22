import 'package:calendar_scheduler/component/main_calendar.dart';
import 'package:calendar_scheduler/component/schedule_bottom_sheet.dart';
import 'package:calendar_scheduler/component/schedule_card.dart';
import 'package:calendar_scheduler/component/today_banner.dart';
import 'package:calendar_scheduler/const/colors.dart';
import 'package:calendar_scheduler/database/drift_database.dart';
import 'package:flutter/material.dart';
import 'package:get_it/get_it.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  // 선택된 날짜를 관리할 변수, 초기값은 오늘 날짜(DateTime.now())를 UTC 기준으로 설정
  DateTime selectedDate = DateTime.utc(
    DateTime.now().year,
    DateTime.now().month,
    DateTime.now().day,
  );

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // HomeScreen에서 FloatingActionButtom을 누르면 ScheduleBottomSheet 화면 추가
      floatingActionButton: FloatingActionButton( // 새 일정 버튼 추가
        backgroundColor: PRIMARY_COLOR,
        onPressed: () {
          showModalBottomSheet( // ScheduleBottomSheet 열기
            context: context, // BuildContext를 전달, 메인 위젯이 지금 트리의 어디에 있는지 알려주는 좌표 같은 역할
            isDismissible: true, // 배경 탭 했을시 ScheduleBottomSheet 닫기
            builder: (_) => ScheduleBottomSheet(
              selectedDate: selectedDate, // 선택된 날짜 "selectedDate" 넘겨주기
            ), // ScheduleBottomSheet() 불러오기
            // ScheduleBottomSheet의 높이를 화면의 최대 높이로 정의 스크롤 가능하게 변경
            isScrollControlled: true,
          );
        },
        child: Icon(
          Icons.add,
        ),
      ),
      body: SafeArea(
        child: Column(
          children: [
            MainCalendar(
              selectedDate: selectedDate, // 선택된 날짜 전달
              onDaySelected: onDaySelected, // 날짜가 선택시 실행할 함수
            ),
            SizedBox(height: 8.0),
            // 일정 Stream으로 받아오기
            StreamBuilder<List<Schedule>>(
              stream: GetIt.I<LocalDatabase>().watchSchedules(selectedDate),
              builder: (context, snapshot) {
                return TodayBanner(
                  selectedDate: selectedDate, 
                  count: snapshot.data?.length ?? 0, // 일정 개수 추가, null 아니면 length 호출/ null이면 0 사용
                );
              },
            ),
            SizedBox(height: 8.0),
            Expanded( // 남은 공간을 모두 차지하기
              // StreamBuilder 비동기 데이터 스트림을 받는다, List<Schedule>로 묶어서 Stream으로 반환
              child: StreamBuilder<List<Schedule>>(
                // GetIt 서비스 로케이터 패턴 라이브러리, 앱 어디서든 LocalDatabase 인스턴스를 가져올수 있다, DB 객체를 전역관리
                // watchSchedules(selectedDate) 메서드는 특정 날짜의 일정 데이터를 Stream 형태로 반환
                stream: GetIt.I<LocalDatabase>().watchSchedules(selectedDate),
                // context 플러터 위젯 트리에서 현재 위치 나타태는 좌표 같은 역할
                // snapshot Stream에서 전달된 현재 상태와 데이터를 담고 있는 객체
                builder: (context, snapshot) {
                  if (!snapshot.hasData) { // 상태 체크, 데이터 없을때
                    return Container();
                  }
                  // 화면에 보이는 값들만 랜더링하는 리스트
                  return ListView.builder(
                    itemCount: snapshot.data!.length, // 리스트에 입력할 값들의 총 개수
                    itemBuilder: (context, index) {
                      // snapshot.data! 는 List<Schedule> 이며, index 번째 일정을 꺼내서 schedule 변수에 저장
                      final schedule = snapshot.data![index];
                      // Dismissible 위젯은 리스트 아이템을 스와이프(밀기)해서 삭제하거나 다른 동작을 실행
                      return Dismissible(
                        key: ObjectKey(schedule.id), // 유니크한 키값
                        direction: DismissDirection.startToEnd, // 밀기 했을때 실행할 함수, 왼쪽->오른쪽으로 밀었을때 작동
                        onDismissed: (DismissDirection direction) { // 아이템이 밀려서 사라질때 실행되는 콜백 함수
                          GetIt.I<LocalDatabase>().removeSchedule(schedule.id); // removeSchedule 삭제 메서드 실행
                        },
                        child: Padding(
                          padding: const EdgeInsets.only(bottom: 8.0, left: 8.0, right: 8.0),
                          child: ScheduleCard(
                            startTime: schedule.startTime, 
                            endTime: schedule.endTime, 
                            content: schedule.content,
                          ),
                        ),
                      );
                    }
                  );
                },
              )
            ),
          ],
        ),
      ),
    );
  }

  // 날짜 선택시 마다 실행할 함수,TableCalendar 파라미터 선택날짜(selectedDate), 기준날짜(focusedDate)
  void onDaySelected(DateTime selectedDate, DateTime focusedDate) {
    // 날짜 업데이트
    setState(() {
      this.selectedDate = selectedDate;
    });
  }
}