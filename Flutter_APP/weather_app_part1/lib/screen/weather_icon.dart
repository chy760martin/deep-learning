import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart'; // SVG 이이지 사용

class WeatherIcon {
  // Widget? 타입인 이유는 각 날씨의 상태를 수치화(id 항목 값)해서 그 수치에 맞는 위젯을 반환 해줘야한다
  // 네트워크 등의 문제로 아이콘이 전달되지 않을 수도 있어 Nullable 타입으로 지정
  Widget? getWeatherIcon(int condition) {
    if (condition < 300) {
      // 천둥 번개를 동반한 강한 비
      return SvgPicture.asset(
        'svg/cloud_lightning.svg',
        colorFilter: const ColorFilter.mode(Colors.black87, BlendMode.srcIn),
      );
    } else if (condition < 600) {
      // 가는 비와 일반적인 비
      return SvgPicture.asset(
        'svg/cloud_rain.svg',
        colorFilter: const ColorFilter.mode(Colors.black87, BlendMode.srcIn),
      );
    } else if (condition < 700) {
      // 흐리고 눈
      return SvgPicture.asset(
        'svg/cloud_snow_alt.svg',
        colorFilter: const ColorFilter.mode(Colors.black87, BlendMode.srcIn),
      );
    } else if (condition == 800) {
      // 맑은 하늘
      return SvgPicture.asset(
        'svg/sun.svg',
        colorFilter: const ColorFilter.mode(Colors.black87, BlendMode.srcIn),
      );
    } else if (condition <= 804) {
      // 구름 낀 하늘
      return SvgPicture.asset(
        'svg/cloud_sun.svg',
        colorFilter: const ColorFilter.mode(Colors.black87, BlendMode.srcIn),
      );
    }
    // 그 외 해당 사항이 없을땐 null 값 반환
    return null;
  }
}