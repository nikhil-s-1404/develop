import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { DomSanitizer } from '@angular/platform-browser';

@Component({
  selector: 'image-upload',
  templateUrl: './image-upload.component.html',
  styleUrls: ['./image-upload.component.scss']
})
export class ImageUploadComponent implements OnInit {
  imageUrl: any;
  box: any;
  annotations: any[] = [];

  constructor(private http: HttpClient, private sanitizer: DomSanitizer) {}
  ngOnInit(): void {
    // throw new Error('Method not implemented.');
  }

  onFileSelected(event: any) {
    const file = event.target.files[0];

    if (file) {
      const blobUrl = URL.createObjectURL(file);
    this.imageUrl = this.sanitizer.bypassSecurityTrustUrl(blobUrl);

      const formData = new FormData();
      formData.append('file', file);

      this.http.post('http://localhost:8080/upload/', formData).subscribe((res: any) => {
        this.annotations = res.annotations;
      });
    } else {
      console.error('No file selected!');
    }
  }


}
