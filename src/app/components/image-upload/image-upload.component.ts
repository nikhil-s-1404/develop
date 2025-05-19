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
  imageLoader = false;

patientName: string = '';
patientAge: number = 0;
patientGender: string = '';


  constructor(private http: HttpClient, private sanitizer: DomSanitizer) { }
  ngOnInit(): void {
    // throw new Error('Method not implemented.');
    this.imageLoader = false;
  }

  predictImage(file: File): Observable<any> {
    const formData: FormData = new FormData();
    formData.append('file', file, file.name);

  formData.append('patient_name', this.patientName);
  formData.append('age', this.patientAge.toString());
  formData.append('gender', this.patientGender);

    // Sending POST request with image as form data
    return this.http.post(this.apiUrl, formData);
  }

  onFileSelected(event: any) {
    const file = event.target.files[0];
    this.imageLoader = false;
    if (file) {
      const blobUrl = URL.createObjectURL(file);
      this.imageUrl = this.sanitizer.bypassSecurityTrustUrl(blobUrl);
      this.imageLoader = true;
      this.annotatedImage = null;
      this.predictImage(file).subscribe({
        next: (response) => {
          this.imageLoader = false;
          console.log('Prediction response:', response);
          let result = JSON.parse(response);
          this.annotatedImage = result.annotated_image; // Store the base64 string
          console.log('final response:', this.annotatedImage);
          console.log('wholesum result:', result);

        },
        error: (error) => {
          this.imageLoader = false;
          console.error('Error predicting image:', error);
        }
      }) // Assuming the response contains the annotated image URL
    } else {
      this.imageLoader = false;
      console.error('No file selected!');
    }
  }


}
