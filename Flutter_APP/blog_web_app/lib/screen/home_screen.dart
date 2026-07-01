import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';

class HomeScreen extends StatefulWidget {
    const HomeScreen({super.key});

    @override
    _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
    // WebViewController를 저장할 변수 선언
    WebViewController? controller;

    @override
    Widget build(BuildContext context) {
        return Scaffold(
            appBar: AppBar(
                backgroundColor: Colors.orange,
                title: const Text('Blog Web App'),
                centerTitle: true,
                // AppBar의 actions 매개 변수
                actions: [
                    // home 아이콘 버튼
                    IconButton(
                        icon: const Icon(Icons.home),
                        // 눌렀을때 콜백 함수 설정
                        onPressed: () {
                        if (controller != null) {
                            // 웹뷰에서 보여줄 사이트 실행하기
                            controller!.loadUrl('https://blog.codefactory.ai'); // Replace with your desired URL
                        }
                        },
                    ),
                ],
            ),
            body: WebView(
                initialUrl: 'https://blog.codefactory.ai', // Replace with your desired URL
                javascriptMode: JavascriptMode.unrestricted,
                
                // 웹뷰 실행 함수
                onWebViewCreated: (WebViewController controller) {
                this.controller = controller;
                },
            ),
            // 네비게이션 바
            bottomNavigationBar: BottomAppBar(
                color: Colors.orange.shade100,
                child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                        IconButton(
                            icon: const Icon(Icons.home),
                            onPressed: () {
                                if (controller != null) {
                                    controller!.loadUrl('https://blog.codefactory.ai'); // Replace with your desired URL
                                }
                            },
                        ),
                        IconButton(
                            icon: const Icon(Icons.arrow_back),
                            onPressed: () async {
                                if (await controller?.canGoBack() ?? false) {
                                    controller!.goBack();
                                }
                            },
                        ),
                        IconButton(
                            icon: const Icon(Icons.arrow_forward),
                            onPressed: () async {
                                if (await controller?.canGoForward() ?? false) {
                                    controller!.goForward();
                                }
                            },
                        ),
                        IconButton(
                            icon: const Icon(Icons.refresh),
                            onPressed: () {
                                if (controller != null) {
                                    controller!.reload();
                                }
                            },
                        ),
                    ],
                ),
            ),
        );
    }
}