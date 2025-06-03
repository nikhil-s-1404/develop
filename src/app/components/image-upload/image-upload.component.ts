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

  frontalFile: File | null = null;
lateralFile: File | null = null;

indication: string = 'indication';
technique: string = 'technique';
comparison: string = 'comparison';

patientName: string = '';
patientAge: number = 0;
patientGender: string = '';

showReportPopup = false;
patientSummary = '';


  constructor(private http: HttpClient, private sanitizer: DomSanitizer) { }
  ngOnInit(): void {
    // throw new Error('Method not implemented.');
    this.imageLoader = false;
  }

  // predictImage(file: File): Observable<any> {
  //   const formData: FormData = new FormData();
  //   formData.append('file', file, file.name);

  // formData.append('patient_name', this.patientName);
  // formData.append('age', this.patientAge.toString());
  // formData.append('gender', this.patientGender);

  //   // Sending POST request with image as form data
  //   return this.http.post(this.apiUrl, formData);
  // }

 onFileSelected(event: any, type: 'frontal' | 'lateral') {
  const file = event.target.files[0];
  if (file) {
    if (type === 'frontal') {
      this.frontalFile = file;
      this.imageUrl = this.sanitizer.bypassSecurityTrustUrl(URL.createObjectURL(file));
    } else {
      this.lateralFile = file;
    }
  }
}

submitImages() {
  if (!this.frontalFile && !this.lateralFile) {
    alert('Please upload at least one image.');
    return;
  }

  const formData = new FormData();
  if (this.frontalFile) formData.append('frontal_image', this.frontalFile);
  if (this.lateralFile) formData.append('lateral_image', this.lateralFile);

  formData.append('indication', this.indication);
  formData.append('technique', this.technique );
  formData.append('comparison', this.comparison );
  // formData.append('file', file, file.name);

  this.imageLoader = true;
  this.annotatedImage = null;

  this.http.post(this.apiUrl, formData).subscribe({
    next: (response: any) => {
      this.imageLoader = false;
      console.log('Prediction response:', response);
      this.annotatedImage = response.annotated_image || null;
      this.patientSummary = response.summary_text;
    },
    error: (err) => {
      this.imageLoader = false;
      console.error('Prediction failed:', err);
    }
  });
}

closePopup() {
  this.showReportPopup = false;
}

showReport() {
  this.showReportPopup = true;
}



}
