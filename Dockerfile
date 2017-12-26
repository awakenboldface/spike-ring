FROM java:8

COPY target/uberjar/spike-ring.jar /code/spike-ring.jar

EXPOSE 8080

CMD ["java", "-jar", "/code/spike-ring.jar"]
