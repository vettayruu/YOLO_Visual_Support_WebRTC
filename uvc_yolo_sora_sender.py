import argparse
import json
import os
import sys
import time
import threading
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

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

        # 各目の最新フレーム
        self.latest_left_frame = None
        self.latest_right_frame = None
        self.frame_lock = threading.Lock()

        # 各目のSora接続
        self.left_sendonly = None
        self.right_sendonly = None

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
        self.capture_thread = threading.Thread(target=self._uvc_capture_loop, daemon=True)
        self.capture_thread.start()
        print("ZED両目映像キャプチャを開始しました")

    def _uvc_capture_loop(self):
        cap = cv2.VideoCapture(0)
        if cap.isOpened() == 0:
            exit(-1)

        # Set the video resolution to HD720 (2560*720)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            # Get a new frame from camera
            retval, frame = cap.read()
            # Extract left and right images from side-by-side
            left_right_image = np.split(frame, 2, axis=1)

            left_image = left_right_image[0]
            right_image = left_right_image[1]

            with self.frame_lock:
                self.latest_left_frame = left_image.copy()
                self.latest_right_frame = right_image.copy()


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
        model_path_seg = "./yolo/models/seg/best.pt"
        # yolo_model = YOLO(model_path_seg)

        yolo_seg_left = YOLOSegPose(model_path_seg)
        yolo_seg_right = YOLOSegPose(model_path_seg)

        try:
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

                    if sent_frames % 100 == 0:
                        elapsed = time.time() - start_time
                        print(f"ステレオ送信中... {elapsed:.1f}秒経過 (送信フレーム数: {sent_frames})")

                        # Save image
                        timestamp = int(time.time() * 1000)
                        cv2.imwrite(f'./record/left/zed_left_{timestamp}.jpg', left_frame)
                        cv2.imwrite(f'./record/right/zed_right_{timestamp}.jpg', right_frame)

                    yolo_seg_left.visualize(left_frame)
                    yolo_seg_right.visualize(right_frame)

                    timestamp = int(time.time() * 1000)
                    print("timestamp:", timestamp)

                    # object
                    pack_cx_l, pack_cy_l, pack_theta_l = yolo_seg_left.get_pack_pose()
                    pack_cx_r, pack_cy_r, pack_theta_r = yolo_seg_right.get_pack_pose()
                    print(f"pack_pose_l: {pack_cx_l}, {pack_cy_l}, {np.rad2deg(pack_theta_l)}")
                    print(f"pack_pose_r: {pack_cx_r}, {pack_cy_r}, {np.rad2deg(pack_theta_r)}")

                    # left_arm
                    left_arm_cx_l, left_arm_cy_l, left_arm_theta_l = yolo_seg_left.get_tip_pose_left()
                    left_arm_cx_r, left_arm_cy_r, left_arm_theta_r = yolo_seg_right.get_tip_pose_left()
                    print(f"left_arm_pose_l: {left_arm_cx_l}, {left_arm_cy_l}, {np.rad2deg(left_arm_theta_l)}")
                    print(f"left_arm_pose_r: {left_arm_cx_r}, {left_arm_cy_r}, {np.rad2deg(left_arm_theta_r)}")

                    # right_arm
                    right_arm_cx_l, right_arm_cy_l, right_arm_theta_l = yolo_seg_left.get_tip_pose_right()
                    right_arm_cx_r, right_arm_cy_r, right_arm_theta_r = yolo_seg_right.get_tip_pose_right()
                    print(f"right_arm_pose_l: {right_arm_cx_l}, {right_arm_cy_l}, {np.rad2deg(right_arm_theta_l)}")
                    print(f"right_arm_pose_r: {right_arm_cx_r}, {right_arm_cy_r}, {np.rad2deg(right_arm_theta_r)}")

                    diff_pose_l = np.rad2deg(right_arm_theta_l) - np.rad2deg(pack_theta_l)
                    diff_pose_r = np.rad2deg(right_arm_theta_r) - np.rad2deg(pack_theta_r)
                    print(f"diff_pose_l: {diff_pose_l} degree")
                    print(f"diff_pose_r: {diff_pose_r} degree")

                    # results = yolo_model(left_frame, verbose=False)
                    # yolo_frame = results[0].plot()

                    if sent_frames % 1000 == 0:
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