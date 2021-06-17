import time
import gcapi

from pathlib import Path
from typing import Iterable


class ThrottledRetries:  # implements an exponential backoff strategy
    def __init__(self, attempts: int, interval: int):
        self.attempts = attempts
        self.interval = interval

    def __iter__(self) -> Iterable[int]:
        for n in range(self.attempts):
            if n > 0:
                time.sleep(min(self.interval * 1.5 ** n, 300))
            yield n


class GrandChallengeAlgorithm:
    def __init__(self, api_token: str, algorithm: str):
        self.algorithm = algorithm
        self.retries = ThrottledRetries(attempts=25, interval=15)
        self.client = gcapi.Client(api_token)

        # Query API to get average runtime
        algorithm_details = self.client(path="algorithms", params={"slug": algorithm})
        try:
            algorithm_details = algorithm_details["results"][0]
        except (KeyError, IndexError):
            raise ValueError("Invalid algorithm name")
        self.algorithm_uuid = algorithm_details["pk"]

        average_duration = algorithm_details["average_duration"]
        if average_duration:
            self.headstart = int(average_duration * 0.75)
        else:
            self.headstart = 30  # default to 30 seconds

    def run(self, input_files: Iterable[str], output_file: str):
        """Uploads the image to grand challenge and downloads the resulting segmentation mask"""
        # Upload to GC
        print(f'Uploading to grand-challenge.org to run "{self.algorithm}"')
        session = self.client.upload_cases(
            files=list(input_files), algorithm=self.algorithm
        )

        # Wait for image import to complete
        print(f'Waiting for image import to finish for algorithm "{self.algorithm}"')
        time.sleep(30)  # give the image import a fixed headstart

        image_uuid = None
        for _ in self.retries:
            try:
                session_details = self.client.raw_image_upload_sessions.detail(
                    session["pk"]
                )
            except IOError:
                continue

            status = session_details["status"]
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f'Algorithm "{self.algorithm}" could not be started')
            elif status == "Succeeded":
                try:
                    image_url = session_details["image_set"][0]
                    image_details = self.client(url=image_url)
                    image_uuid = image_details["pk"]
                except IndexError:
                    raise RuntimeError("Uploaded image could not be imported, not a valid image?")
                except IOError:
                    continue
                else:
                    break
        else:
            raise TimeoutError

        # Wait for job to complete
        print(f'Waiting for results of algorithm "{self.algorithm}"')
        time.sleep(self.headstart)

        query_params = {
            "algorithm_image__algorithm": self.algorithm_uuid,
            "input_image": image_uuid
        }
        for _ in self.retries:
            try:
                algorithm_job_details = self.client.algorithm_jobs.page(params=query_params, limit=1)[0]
            except (IOError, IndexError):
                continue

            status = algorithm_job_details["status"]
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f'Algorithm "{self.algorithm}" failed')
            elif status == "Succeeded":
                print(f'Downloading results of algorithm "{self.algorithm}"')
                self._download_results(algorithm_job_details["outputs"], output_file)
                break
        else:
            raise TimeoutError

    def _download_results(self, outputs: Iterable, filename: str):
        for output in outputs:
            if output["image"] is None:
                continue

            image_details = self.client(url=output["image"])
            for file in image_details["files"]:
                # Skip images that are not mha files
                if file["image_type"] != "MHD":
                    continue

                # Download data and dump into file
                output_file = Path(filename)
                if output_file.suffix != ".mha":
                    raise ValueError("Output file needs to have .mha extension")

                output_file.parent.mkdir(parents=True, exist_ok=True)
                with output_file.open("wb") as fp:
                    fp.write(self.client(url=file["file"]).content)

                return  # there is only one mask that we need to save
