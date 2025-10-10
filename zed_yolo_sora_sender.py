#!/usr/bin/env python3
"""
ZED Miniの左目・右目映像をSoraに送信するスクリプト

使用方法:
    # 左目のみ送信
    python zed_sora_sender.py --eye left

    # 右目のみ送信
    python zed_sora_sender.py --eye right

    # 両目同時送信（推奨）
    python zed_sora_sender.py --eye both

必要な環境変数:
    SORA_SIGNALING_URLS: SoraサーバーのWebSocketdef main():
    # 引数解析
    parser = argparse.ArgumentParser(description="ZED Mini映像をSoraに送信")
    parser.add_argument("--eye", "-e", choices=["left", "right", "both"], default="both",
                       help="送信する目 (デフォルト: both)")
    parser.add_argument("--left-channel", "-l", default="sora_liust_left",
                       help="左目チャンネルID (デフォルト: sora_liust_left)")
    parser.add_argument("--right-channel", "-r", default="sora_liust_right",
                       help="右目チャンネルID (デフォルト: sora_liust_right)")
    parser.add_argument("--duration", "-d", type=int, default=0, help="送信時間（秒、0で無制限）")
    parser.add_argument("--resolution", "-res", choices=["HD720", "HD1080", "VGA"],
                       default="HD720", help="ZED解像度")
    parser.add_argument("--fps", "-f", type=int, default=30, help="ZED FPS")ra2.uclab.jp)
    SORA_METADATA: 認証情報を含むJSONメタデータ
"""

import argparse
import json
import os
import sys
import time
import threading
from typing import Optional

import pyzed.sl as sl
import cv2
import numpy as np
from dotenv import load_dotenv
from YOLOSegPose import YOLOSegPose

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from media_sendonly import Sendonly

class ZEDStereoSender:
    """ZED Miniの両目映像をSoraに同時送信するクラス"""

    def __init__(self, signaling_urls: list[str], left_channel: str, right_channel: str, metadata: dict):
        self.signaling_urls = signaling_urls
        self.left_channel = left_channel
        self.right_channel = right_channel
        self.metadata = metadata

        # ZED初期化
        self.zed = sl.Camera()
        self.image_left = sl.Mat()
        self.image_right = sl.Mat()
        self.running = False

        # 各目の最新フレーム
        self.latest_left_frame = None
        self.latest_right_frame = None
        self.frame_lock = threading.Lock()

        # 各目のSora接続
        self.left_sendonly = None
        self.right_sendonly = None

    def initialize_zed(self, resolution=sl.RESOLUTION.HD720, fps=60):
        """ZEDカメラを初期化"""
        print("ZEDカメラを初期化中...")
        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution
        init_params.camera_fps = fps
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        init_params.coordinate_units = sl.UNIT.METER

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"ZED初期化エラー: {err}")
            return False

        cam_info = self.zed.get_camera_information()
        print(f"ZEDカメラ接続成功")
        print(f"  シリアル番号: {cam_info.serial_number}")
        print(f"  解像度: {resolution}")
        print(f"  FPS: {fps}")
        return True

    def initialize_sora_connections(self):
        """左目と右目のSora接続を初期化"""
        print("Sora接続を初期化中...")

        # 左目接続
        self.left_sendonly = Sendonly(
            signaling_urls=self.signaling_urls,
            channel_id=self.left_channel,
            metadata=self.metadata,
            video=True,
            audio=False,
            video_codec_type="VP8",
            video_bit_rate=5000,
        )

        # 右目接続
        self.right_sendonly = Sendonly(
            signaling_urls=self.signaling_urls,
            channel_id=self.right_channel,
            metadata=self.metadata,
            video=True,
            audio=False,
            video_codec_type="VP8",
            video_bit_rate=5000,
        )

        print(f"  左目チャンネル: {self.left_channel}")
        print(f"  右目チャンネル: {self.right_channel}")

    def start_capture(self):
        """ZED映像キャプチャを開始"""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("ZED両目映像キャプチャを開始しました")

    def _capture_loop(self):
        """映像キャプチャのメインループ"""
        runtime = sl.RuntimeParameters()
        frame_count = 0

        while self.running:
            if self.zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                # 左目と右目の両方を取得
                self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
                self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)

                left_image = self.image_left.get_data()
                right_image = self.image_right.get_data()

                # BGRAからBGRに変換
                if len(left_image.shape) == 3 and left_image.shape[2] == 4:
                    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGRA2BGR)
                if len(right_image.shape) == 3 and right_image.shape[2] == 4:
                    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGRA2BGR)

                with self.frame_lock:
                    self.latest_left_frame = left_image.copy()
                    self.latest_right_frame = right_image.copy()

                # frame_count += 1
                # if frame_count % 300 == 0:  # 10秒ごとに状況報告
                #     print(f"ZED両目フレーム取得中... ({frame_count}フレーム)")
            else:
                time.sleep(0.01)

    def get_frames(self):
        """最新の左目・右目フレームを取得"""
        with self.frame_lock:
            left_frame = self.latest_left_frame.copy() if self.latest_left_frame is not None else None
            right_frame = self.latest_right_frame.copy() if self.latest_right_frame is not None else None
            return left_frame, right_frame

    def run(self, duration_seconds=0):
        """両目映像送信を実行"""
        print("ZED ステレオ映像送信を開始します")
        print(f"左目チャンネル: {self.left_channel}")
        print(f"右目チャンネル: {self.right_channel}")

        if duration_seconds == 0:
            print("送信時間: 無制限 (Ctrl+Cで停止)")
        else:
            print(f"送信時間: {duration_seconds}秒")

        print("=" * 60)
        
        model_path_seg = "./model/best.pt"
        yolo_seg_left = YOLOSegPose(model_path_seg)
        yolo_seg_right = YOLOSegPose(model_path_seg)

        try:
            # ZED初期化
            if not self.initialize_zed():
                return False

            # Sora接続初期化
            self.initialize_sora_connections()

            # キャプチャ開始
            self.start_capture()
            time.sleep(1)  # フレームが準備されるまで待機

            # 両方のSora接続を開始
            print("Soraサーバーに接続中...")
            self.left_sendonly.connect()
            self.right_sendonly.connect()

            print("両目映像送信を開始します")
            print("=" * 50)
            print(f"左目チャンネル: {self.left_channel}")
            print(f"右目チャンネル: {self.right_channel}")
            print("=" * 50)

            # 映像送信ループ
            start_time = time.time()
            sent_frames = 0

            if duration_seconds == 0:
                # 無制限送信
                while True:
                    left_frame, right_frame = self.get_frames()
                    sent_frames += 1

                    if sent_frames % 500 == 0:
                        elapsed = time.time() - start_time
                        print(f"ステレオ送信中... {elapsed:.1f}秒経過 (送信フレーム数: {sent_frames})")

                        # Save image
                        timestamp = int(time.time() * 1000)
                        cv2.imwrite(f'./record/left/zed_left_{timestamp}.jpg', left_frame)
                        cv2.imwrite(f'./record/right/zed_right_{timestamp}.jpg', right_frame)

                    yolo_seg_left.visualize(left_frame)
                    yolo_seg_right.visualize(right_frame)

                    if sent_frames % 500 == 0:
                        # Save image
                        timestamp = int(time.time() * 1000)
                        cv2.imwrite(f'./record/yolo_left/yolo_left_{timestamp}.jpg', left_frame)
                        cv2.imwrite(f'./record/yolo_right/yolo_right_{timestamp}.jpg', right_frame)

                    if left_frame is not None and right_frame is not None:
                        self.left_sendonly._video_source.on_captured(left_frame)
                        self.right_sendonly._video_source.on_captured(right_frame)

                    # time.sleep(0.01667)  # MAX 100FPS
            else:
                # 指定時間送信
                while time.time() - start_time < duration_seconds:
                    left_frame, right_frame = self.get_frames()

                    if left_frame is not None and right_frame is not None:
                        self.left_sendonly._video_source.on_captured(left_frame)
                        self.right_sendonly._video_source.on_captured(right_frame)
                        sent_frames += 1

                        # 5秒ごとに進捗表示
                        if sent_frames % 500 == 0:
                            elapsed = time.time() - start_time
                            # yolo_pose.record(left_frame, right_frame)
                            print(f"ステレオ送信中... {elapsed:.1f}秒経過 (送信フレーム数: {sent_frames})")

                    time.sleep(0.008)  # MAX 120FPS

            print(f"ステレオ映像送信完了! 総送信フレーム数: {sent_frames}")

        except KeyboardInterrupt:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            print(f"\nユーザーによって中断されました")
            print(f"送信時間: {hours:02d}:{minutes:02d}:{seconds:02d}")
            print(f"総送信フレーム数: {sent_frames if 'sent_frames' in locals() else 0}")
        except Exception as e:
            print(f"エラーが発生しました: {e}")
        finally:
            self.cleanup()

        return True

    def cleanup(self):
        """リソースのクリーンアップ"""
        print("ステレオ送信クリーンアップ中...")

        if self.left_sendonly:
            self.left_sendonly.disconnect()
        if self.right_sendonly:
            self.right_sendonly.disconnect()

        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=5)

        self.zed.close()
        print("ステレオ送信クリーンアップ完了")


def main():
    # 引数解析
    parser = argparse.ArgumentParser(description="ZED Mini映像をSoraに送信")
    parser.add_argument("--eye", "-e", choices=["left", "right", "both"], default="both",
                        help="送信する目 (デフォルト: both)")
    parser.add_argument("--left-channel", "-l", default="sora_liust_left",
                        help="左目チャンネルID (デフォルト: sora_liust_left)")
    parser.add_argument("--right-channel", "-r", default="sora_liust_right",
                        help="右目チャンネルID (デフォルト: sora_liust_right)")
    parser.add_argument("--duration", "-d", type=int, default=0, help="送信時間（秒、0で無制限）")
    parser.add_argument("--resolution", "-res", choices=["HD720", "HD1080", "VGA"],
                        default="HD1080", help="ZED解像度")
    parser.add_argument("--fps", "-f", type=int, default=30, help="ZED FPS")

    args = parser.parse_args()

    # 環境変数読み込み
    load_dotenv()

    # 必要な環境変数をチェック
    signaling_urls_str = os.getenv("SORA_SIGNALING_URLS")
    if not signaling_urls_str:
        print("エラー: 環境変数 SORA_SIGNALING_URLS が設定されていません")
        sys.exit(1)

    metadata_str = os.getenv("SORA_METADATA")
    if not metadata_str:
        print("エラー: 環境変数 SORA_METADATA が設定されていません")
        sys.exit(1)

    # 設定解析
    signaling_urls = [url.strip() for url in signaling_urls_str.split(",")]
    metadata = json.loads(metadata_str)

    # 解像度設定
    resolution_map = {
        "HD720": sl.RESOLUTION.HD720,
        "HD1080": sl.RESOLUTION.HD1080,
        "VGA": sl.RESOLUTION.VGA
    }
    resolution = resolution_map[args.resolution]

    print("ZED Sora映像送信を開始します")
    print(f"送信モード: {args.eye}")
    if args.duration == 0:
        print("送信時間: 無制限 (Ctrl+Cで停止)")
    else:
        print(f"送信時間: {args.duration}秒")
    print(f"解像度: {args.resolution}")
    print(f"FPS: {args.fps}")
    print()

    # 送信開始
    if args.eye == "both":
        # 両目同時送信
        stereo_sender = ZEDStereoSender(signaling_urls, args.left_channel, args.right_channel, metadata)
        stereo_sender.run(args.duration)

if __name__ == "__main__":

    main()
