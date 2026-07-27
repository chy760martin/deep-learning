import 'package:geolocator/geolocator.dart';

class MyLocation {
  double? myLatitude;
  double? myLongitude;

  Future<void> getMyCurrentLocation() async {
    // getlocator 기능 구현
    try {
      LocationPermission permission = await Geolocator.requestPermission();
      Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high
      );
      myLatitude = position.latitude;
      myLongitude = position.longitude;

      print(position);
      print('Latitude: ${myLatitude}');
      print('Logitude: ${myLongitude}');
    } catch (e) {
      print("위치 정보 수신에 문제가 생겼습니다.: ${e}");
    }
  }
}