import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { DomSanitizer } from '@angular/platform-browser';
import { Observable } from 'rxjs';

@Component({
  selector: 'image-upload',
  templateUrl: './image-upload.component.html',
  styleUrls: ['./image-upload.component.scss']
})
export class ImageUploadComponent implements OnInit {
  imageUrl: any;
  box: any;
  annotations: any[] = [];
  annotatedImage: string | null = null; // Store the annotated image as base64 string
  apiUrl = 'http://localhost:8080/predict/';

  constructor(private http: HttpClient, private sanitizer: DomSanitizer) {}
  ngOnInit(): void {
    // throw new Error('Method not implemented.');
  }

  predictImage(file: File): Observable<any> {
    const formData: FormData = new FormData();
    formData.append('file', file, file.name);

    // Sending POST request with image as form data
    return this.http.post(this.apiUrl, formData);
  }

  onFileSelected(event: any) {
    const file = event.target.files[0];

    if (file) {

    this.predictImage(file).subscribe({
      next: (response) => {
        console.log('Prediction response:', response);
        let result = JSON.parse(response);
        this.annotatedImage = result.annotated_image; // Store the base64 string
        console.log('final response:', this.annotatedImage);

      },
      error: (error) => {
        console.error('Error predicting image:', error);
      }
    }) // Assuming the response contains the annotated image URL
    } else {
      console.error('No file selected!');
    }
  }


}
